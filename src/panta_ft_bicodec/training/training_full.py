""" Full fine-tuning of BiCodec: encoder, quantizer, speaker_encoder, prenet and decoder are trained """
import argparse
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from panta_ft_bicodec.data.dataset import CustomDatasetAudio
from panta_ft_bicodec.model.bicodec_tokenizer import BiCodecTokenizer
from panta_ft_bicodec.model.disciminator import Discriminator
from panta_ft_bicodec.read_config import read_config
from panta_ft_bicodec.training.loss import GANLoss, MelSpectrogramLoss, compute_loss_discriminative, compute_loss_gen, eval_model
from panta_ft_bicodec.training.utils import CosineSchedulerWithInternalState, RunningMetrics, set_seed
import logging
import mlflow
from safetensors.torch import save_file
from dataclasses import asdict
import torch.distributed as dist
from panta_ft_bicodec.training.utils_multi_gpu import save_optimizers_and_steps, setup_ddp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("training_full.log", mode="w"),  # "w" = overwrite
        logging.StreamHandler()  # console
    ]
)

def build_folder_to_save(config: dict) -> str:
    name = "full_lr_" + str(config["training"]["lr"]) + "_lr_ad" + str(config["adversarial_model"]["lr"]) + "_norm_disc_" + str(int(config["training"]["discriminator_norm_gradient"])) + "_norm_gen_" + str(int(config["training"]["generator_norm_gradient"]))
    return name


def select_trainable_modules(bicodec: nn.Module) -> list[nn.Parameter]:
    """ Choose which parts of BiCodec are trained. Returns the trainable parameters.

    Trained: encoder, quantizer, speaker_encoder, prenet, decoder.
    Frozen: postnet (unused in forward) and the ECAPA x-vector branch (pool, bn, linear, bn2),
    which does not feed d_vector: DDP fails on trainable params that get no gradient.
    """
    bicodec.requires_grad_(False)
    bicodec.encoder.requires_grad_(True)
    bicodec.quantizer.requires_grad_(True)
    bicodec.speaker_encoder.requires_grad_(True)
    bicodec.prenet.requires_grad_(True)
    bicodec.decoder.requires_grad_(True)
    ecapa = bicodec.speaker_encoder.speaker_encoder
    for module in (ecapa.pool, ecapa.bn, ecapa.linear, ecapa.bn2):
        module.requires_grad_(False)
    bicodec.finetune_encoder = True  # forward builds a graph through encoder/quantizer/speaker_encoder
    return [p for p in bicodec.parameters() if p.requires_grad]


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
    if config["training"]["load_model"] is not None:
        logging.info("Load model")
        model.load_trained_model(str(current_path / name_folder_to_save / config["training"]["load_model"]) +".safetensors")
        disciminator.load(str(current_path / name_folder_to_save / config["training"]["load_model"]) +"_disc.safetensors")
    trainable_parameters = select_trainable_modules(model.model)
    if is_main_process:
        logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters)}")

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
    optimizer_generator = torch.optim.AdamW(
        trainable_parameters,
        lr=config["training"]["lr"],
        betas=tuple(config["training"].get("betas", (0.8, 0.9))),
        weight_decay=config["training"].get("weight_decay", 0.01)
    )
    optimizer_discriminator = torch.optim.AdamW(
        disciminator.parameters(),
        lr=config["adversarial_model"]["lr"],
        betas=tuple(config["adversarial_model"].get("betas", (0.8, 0.9))),
        weight_decay=config["adversarial_model"].get("weight_decay", 0.01)
    )
    total_steps = config["training"]["nb_steps"]
    # Phase 1 (steps < disc_pretrain_steps): the pretrained generator is frozen, only D learns,
    # since a randomly initialised D gives a useless signal. Phase 2: GAN, ramped in over gan_ramp_steps.
    disc_pretrain_steps = config["training"].get("disc_pretrain_steps", 20_000)
    gan_ramp_steps = config["training"].get("gan_ramp_steps", 10_000)
    if disc_pretrain_steps >= total_steps:
        raise ValueError("disc_pretrain_steps must be smaller than nb_steps")
    # The generator scheduler only steps in phase 2, so its warmup and cosine start there
    scheduler_generator = CosineSchedulerWithInternalState(
        optimizer_generator,
        t_initial=total_steps - disc_pretrain_steps,
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
    if config["training"]["load_model"] is not None:
        logging.info("load optimizers and schedulers")
        logging.info(f"Loading checkpoint from {config['training']['load_model']}")
        path_to_checkpoint = str(current_path / name_folder_to_save / config["training"]["load_model"])
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
    log_step = config["training"].get("log_step", 100)
    eval_step = config["training"]["eval_step"]
    running_metrics = RunningMetrics()

    def evaluate_and_save(steps: int, epoch: int) -> None:
        """ Evaluate on the validation set, log to MLflow and save a checkpoint (rank 0) """
        logging.info(f" Evaluation at step {steps}")
        val_metric = asdict(eval_model(
            dataloader=dataloader_val,
            model=model,
            mel_loss=mel_loss,
            gan_loss=gan_loss,
            device=device))
        if is_main_process:
            logging.info(f"Validation metrics at step {steps}: {val_metric}")
            mlflow.log_metrics(val_metric, step=steps)
            path_to_save = current_path / name_folder_to_save / f"step_{steps}_val_mel_{val_metric['mel_loss']:.2f}"
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
            disciminator.module.save(f"{path_to_save}_disc.safetensors")
        dist.barrier()

    while steps < config["training"]["nb_steps"]:
        sampler_train.set_epoch(epoch)
        for batch in dataloader_train:
            batch = batch.to(device)
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            train_generator = steps >= disc_pretrain_steps
            with torch.set_grad_enabled(train_generator):
                outputs, vq_metrics = model(batch)

            loss_d = compute_loss_discriminative(
                x=outputs.detach(),
                y=batch,
                gan_loss=gan_loss
            )
            # Update discriminator
            loss_d.backward()
            grad_norm_d = nn.utils.clip_grad_norm_(
                disciminator.parameters(),
                max_norm=config["training"]["discriminator_norm_gradient"]
            )
            optimizer_discriminator.step()
            scheduler_discriminator.step()

            gan_weight = 0.0
            if train_generator:
                gan_weight = max(0.0, min(1.0, (steps - disc_pretrain_steps) / gan_ramp_steps))
                apply_gan = gan_weight > 0.0
                loss_g, loss_g_details = compute_loss_gen(
                    x=outputs,
                    y=batch,
                    mel_loss=mel_loss,
                    gan_loss=gan_loss,
                    apply_gan=apply_gan,
                    weights=gan_weight,
                    return_details=True
                )
                loss_g += vq_metrics["vq_loss"]  # trains encoder (commitment) and codebook
                # Update generator
                loss_g.backward()
                grad_norm_g = nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=config["training"]["generator_norm_gradient"]
                )
                optimizer_generator.step()
                scheduler_generator.step()
            steps += 1
            if is_main_process:
                step_metrics = {
                    "train/loss_d": loss_d.detach(),
                    "train/vq_loss": vq_metrics["vq_loss"].detach(),
                    "train/grad_norm_d": grad_norm_d,
                }
                if train_generator:
                    step_metrics.update({
                        "train/loss_g": loss_g.detach(),
                        "train/grad_norm_g": grad_norm_g,
                        **{f"train/{name}": value.detach() for name, value in loss_g_details.items()},
                    })
                running_metrics.add(step_metrics)
                if steps % log_step == 0:
                    train_metric = running_metrics.pop_means()
                    train_metric["train/gan_weight"] = gan_weight
                    train_metric["train/lr_g"] = optimizer_generator.param_groups[0]["lr"]
                    train_metric["train/lr_d"] = optimizer_discriminator.param_groups[0]["lr"]
                    mlflow.log_metrics(train_metric, step=steps)
            end_of_disc_pretraining = steps == disc_pretrain_steps
            end_of_training = steps >= config["training"]["nb_steps"]
            if steps % eval_step == 0 or end_of_disc_pretraining or end_of_training:
                evaluate_and_save(steps=steps, epoch=epoch)
            if end_of_training:
                break
        epoch += 1
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Full fine-tuning of the BiCodec model")
    argument_parser.add_argument("--config", help="Path to the configuration file")
    args = argument_parser.parse_args()
    current_path = Path(__file__).resolve().parent
    config_path = current_path.parent / "config" / args.config
    config = read_config(str(config_path))
    train(config=config)
