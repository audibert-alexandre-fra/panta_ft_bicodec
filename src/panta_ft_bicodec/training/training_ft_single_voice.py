import argparse
import os
from pathlib import Path
import torch
from torch import device, nn
from torch.utils.data import DataLoader
import tqdm
from panta_ft_bicodec.data.dataset import CustomDatasetAudio
from panta_ft_bicodec.model.bicodec_tokenizer import BiCodecTokenizer
from panta_ft_bicodec.model.disciminator import Discriminator
from panta_ft_bicodec.read_config import read_config
from panta_ft_bicodec.training.loss import GANLoss, MelSpectrogramLoss, compute_loss_discriminative, compute_loss_gen, eval_model
from panta_ft_bicodec.training.utils import CosineSchedulerWithInternalState, set_seed
import logging
import mlflow
from tqdm import tqdm
from safetensors.torch import save_file
from dataclasses import asdict
import torch.distributed as dist
from panta_ft_bicodec.training.utils_multi_gpu import save_optimizers_and_steps, setup_ddp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("training_mutli_gpu_pretokenzie.log", mode="w"),  # "w" = overwrite
        logging.StreamHandler()  # console
    ]
)

def build_folder_to_save(config: dict) -> str:
    name = "ft_lr_" + str(config["training"]["lr"]) + '_steps_' + str(config["training"]["nb_steps"])
    return name 

def train(config: dict):
    """ train the model """
    name_folder_to_save = build_folder_to_save(config)
    os.makedirs(name_folder_to_save, exist_ok=True)
    current_path = Path(__file__).resolve().parent
    device, rank, world_size, local_rank, is_main_process = setup_ddp(type_ddp=config["training"]["type_ddp"])
    logging.info(f"Starting training with config:{local_rank}, {is_main_process}")
    set_seed(seed=config["training"]["seed"])
    logging.info("Logging Dataset")
    dataset_train = CustomDatasetAudio(config["dataset"]["list_path_audio"])
    dataset_val = dataset_train.split_dataset(nb_audios=config["dataset"]["nb_val"])
    logging.info("Build")
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=config["training"]["batch_size"],
        sampler=sampler_train,
        num_workers=4
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    model = BiCodecTokenizer(device=device)
    disciminator = Discriminator()
    disciminator.to(device=device)
    logging.info("Load model")
    model.load_trained_model(str(current_path / "lr_2e-05_lr_ad3e-05_norm_disc_5_norm_gen_10" / "val_loss_51_lr_20.safetensors"))
    disciminator.load(str(current_path / "lr_2e-05_lr_ad3e-05_norm_disc_5_norm_gen_10" / "val_loss_51_lr_20.safetensors_disc.safetensors"))

    dist.barrier()  
    model.model = DDP(
        model.model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    disciminator = DDP(
        disciminator,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    optimizer_generator = torch.optim.AdamW(model.model.module.get_parameter(), lr=config["training"]["lr"], betas=(0.8, 0.9))
    optimizer_discriminator = torch.optim.AdamW(disciminator.parameters(), lr=config["adversarial_model"]["lr"], betas=(0.8, 0.9))
    total_steps = config["training"]["nb_steps"]
    scheduler_generator = CosineSchedulerWithInternalState(
        optimizer_generator, 
        t_initial=total_steps,
        lr_min=config["training"]["min_lr"],
        warmup_t=config["training"]["warmup_step_generator"]
    )

    scheduler_discriminator = CosineSchedulerWithInternalState(
        optimizer_discriminator, 
        t_initial=total_steps,
        lr_min=config["training"]["min_lr"],
        warmup_t=config["training"]["warmup_step_disciminator"]
    )
    mel_loss = MelSpectrogramLoss(weight=config["loss"]["weight_mel"])
    mel_loss.to(device=device)
    gan_loss = GANLoss(discriminator=disciminator)
    steps = 0
    epoch = 0
    val_metric = asdict(
        eval_model(dataloader=dataloader_val,
        model=model,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        device=device)
    )  # Eval avant le début de l'entraînement
    if is_main_process:
        mlflow.set_experiment(config["training"]["experiment_name"])
        mlflow.start_run(run_name=config["training"]["run_name"])
        mlflow.log_dict(config, "config.json")
        logging.info(f"Validation metrics at step {epoch}: {val_metric}")
        mlflow.log_metrics(val_metric, step=steps)
    dist.barrier()
    if is_main_process:
        logging.info(f"Starting training loop with {steps} steps already done.")
    while steps < config["training"]["nb_steps"]:
        sampler_train.set_epoch(epoch)
        for batch in dataloader_train:
            batch = batch.to(device)
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            outputs, vq_metrics = model(batch)

            loss_d = compute_loss_discriminative(
                x=outputs.detach(),
                y=batch,
                gan_loss=gan_loss
            )
            # Update discriminator
            loss_d.backward()
            nn.utils.clip_grad_norm_(
                disciminator.parameters(),
                max_norm=config["training"]["discriminator_norm_gradient"]
            )
            optimizer_discriminator.step()
            scheduler_discriminator.step()
            loss_g = compute_loss_gen(
                x=outputs,
                y=batch,
                mel_loss=mel_loss,
                gan_loss=gan_loss,
                apply_gan=True,
            )
            loss_g += vq_metrics["vq_loss"]
            # Update generator
            loss_g.backward()
            nn.utils.clip_grad_norm_(
                model.model.module.get_parameter(),
                max_norm=config["training"]["generator_norm_gradient"]
            )
            optimizer_generator.step()
            scheduler_generator.step()
            steps += 1
            if steps >= config["training"]["nb_steps"]:
                break
        epoch += 1
        logging.info(f" Evaluation {epoch}")
        val_metric = asdict(eval_model(
            dataloader=dataloader_val,
            model=model,
            mel_loss=mel_loss,
            gan_loss=gan_loss,
            device=device)
        )
        if is_main_process:
            logging.info(f"Validation metrics at step {epoch}: {val_metric}")
            logging.info(f" Steps: {steps}")
        if is_main_process:
            path_to_save_folder = name_folder_to_save
            mlflow.log_metrics(val_metric, step=steps)
            path_to_save = Path(__file__).resolve().parent / path_to_save_folder / f"val_loss_{val_metric['mel_loss']:.0f}_lr_{epoch}"
            path_to_save_disc = str(Path(__file__).resolve().parent / path_to_save_folder / f"val_loss_{val_metric['mel_loss']:.0f}_lr_{epoch}_disc.safetensors")
            state_dict = {
                k: v for k, v in model.model.module.state_dict().items()
                if 'mel_transformer' not in k
            }
            save_file(state_dict,  Path(f"{path_to_save}.safetensors"))
            save_optimizers_and_steps(
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                scheduler_discriminator=scheduler_discriminator,
                scheduler_generator=scheduler_generator,
                epoch=epoch,
                steps=steps,
                checkpoint_path=Path(f"{path_to_save}.pt")
            )
            disciminator.module.save(path_to_save_disc)
        dist.barrier()
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Train the BiCodec model")
    argument_parser.add_argument("--config", help="Path to the configuration file")
    args = argument_parser.parse_args()
    current_path = Path(__file__).resolve().parent
    config_path = current_path.parent / "config" / args.config
    config = read_config(str(config_path))
    train(config=config)
