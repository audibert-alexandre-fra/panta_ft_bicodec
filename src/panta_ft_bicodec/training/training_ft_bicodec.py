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


def train(config: dict):
    """ train the model """
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
    if config["training"]["load_model"] is not None:
        logging.info("Load model")
        model.load_trained_model(str(current_path / "checkpoints" / config["training"]["load_model"]) +".safetensors")
    model.model = DDP(
        model.model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        skip_all_reduce_unused_params=True
    )
    disciminator = Discriminator()
    disciminator.to(device=device)
    disciminator = DDP(
        disciminator,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    optimizer_generator = torch.optim.AdamW(model.model.module.get_parameter_ft_bicodec(), lr=config["training"]["lr"])
    optimizer_discriminator = torch.optim.AdamW(disciminator.parameters(), lr=config["adversarial_model"]["lr"])
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
    mel_loss.set_device(device=device)
    gan_loss = GANLoss(discriminator=disciminator)
    if config["training"]["load_model"] is not None:
        logging.info("load optimizers and schedulers")
        logging.info(f"Loading checkpoint from {config['training']['load_model']}")
        path_to_checkpoint = str(current_path / "checkpoints" / config["training"]["load_model"])
        checkpoint = torch.load(path_to_checkpoint + ".pt", map_location=device)
        optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        scheduler_discriminator.load_state_dict(checkpoint['scheduler_discriminator'])
        scheduler_generator.load_state_dict(checkpoint['scheduler_generator'])
        steps = checkpoint['steps']
        epoch = checkpoint['epoch']
    else:
        logging.info("No checkpoint found, starting normally")
        steps = 0
        epoch = 0
    val_metric = asdict(eval_model(dataloader=dataloader_val, model=model, mel_loss=mel_loss, device=device))  # Eval avant le début de l'entraînement
    print(val_metric)
    if is_main_process:
        mlflow.set_experiment(config["training"]["experiment_name"])
        mlflow.start_run(run_name=config["training"]["run_name"])
        mlflow.log_dict(config, "config.json")
        mlflow.log_metrics(val_metric, step=steps)
    dist.barrier()
    logging.info(f"Starting training loop with {steps} steps already done.")

    scaler_generator = GradScaler('cuda')
    scaler_discriminator = GradScaler('cuda')
    while steps < config["training"]["nb_steps"]:
        sampler_train.set_epoch(epoch)
        for batch in tqdm(dataloader_train, desc="Training"):
            optimizer_discriminator.zero_grad()
            with torch.no_grad():
                with autocast("cuda"):
                    outputs, _ = model(batch)
            with autocast("cuda"):
                loss_d = compute_loss_discriminative(x=outputs, y=batch, gan_loss=gan_loss)
            scaler_discriminator.scale(loss_d).backward()
            scaler_discriminator.unscale_(optimizer_discriminator)
            nn.utils.clip_grad_norm_(disciminator.parameters(), max_norm=1.0)
            scaler_discriminator.step(optimizer_discriminator)
            scaler_discriminator.update()
            scheduler_discriminator.step()
            optimizer_generator.zero_grad()
            with autocast("cuda"):
                outputs, vq_metrics = model(batch)
                loss_g = compute_loss_gen(
                    x=outputs,
                    y=batch,
                    mel_loss=mel_loss,
                    gan_loss=gan_loss,
                    apply_gan=steps > config["training"]["warmup_step_generator"]
                )
                loss_g += vq_metrics["vq_loss"]
            scaler_generator.scale(loss_g).backward()
            scaler_generator.unscale_(optimizer_generator)
            nn.utils.clip_grad_norm_(model.model.module.get_parameter_ft_bicodec(), max_norm=1.0)
            scaler_generator.step(optimizer_generator)
            scaler_generator.update()
            scheduler_generator.step()
            steps += 1
            if steps >= config["training"]["nb_steps"]:
                break
        if is_main_process:
            logging.info(f" Evaluation {epoch}")
            val_metric = asdict(eval_model(dataloader=dataloader_val, model=model, mel_loss=mel_loss, device=device))
            logging.info(f"Validation metrics at step {epoch}: {val_metric}")
            mlflow.log_metrics(val_metric, step=steps)
            path_to_save = Path(__file__).resolve().parent / "checkpoints" / f"val_loss_{val_metric['mel_loss']:.0f}_lr_{epoch}"
            state_dict = {
                k: v for k, v in model.model.state_dict().items()
                if'mel_transformer' not in k
            }
            save_file(state_dict,  Path(f"{path_to_save}.safetensors"))
            epoch += 1
            save_optimizers_and_steps(
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                scheduler_discriminator=scheduler_discriminator,
                scheduler_generator=scheduler_generator,
                epoch=epoch,
                steps=steps,
                checkpoint_path=Path(f"{path_to_save}.pt")
            )
        dist.barrier()
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    current_path = Path(__file__).resolve().parent
    config_path = current_path.parent / "config" / "config.yaml"
    config = read_config(str(config_path))
    train(config=config)