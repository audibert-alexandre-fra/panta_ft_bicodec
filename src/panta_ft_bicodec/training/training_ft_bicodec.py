from pathlib import Path
import torch
from torch import device, nn
from torch.utils.data import DataLoader
import tqdm
from panta_ft_bicodec.constant import SAMPLING_RATE
from panta_ft_bicodec.data.dataset import CustomDatasetAudio
from panta_ft_bicodec.model.bicodec_tokenizer import BiCodecTokenizer
from panta_ft_bicodec.model.disciminator import Discriminator
from panta_ft_bicodec.read_config import read_config
from panta_ft_bicodec.training.loss import GANLoss, MelSpectrogramLoss, compute_loss_discriminative, compute_loss_gen, eval_model
from panta_ft_bicodec.training.utils import CosineSchedulerWithInternalState, get_available_device, set_seed
import logging
import torch.nn.functional as F
import torchaudio.transforms as T
import mlflow
from tqdm import tqdm
from safetensors.torch import load_model, save_model
from safetensors.torch import save_file
from dataclasses import asdict

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
    logging.info("Starting training with config:")
    set_seed(seed=config["training"]["seed"])
    logging.info("Logging Dataset")
    dataset_train = CustomDatasetAudio(config["dataset"]["list_path_audio"])
    dataset_val = dataset_train.split_dataset(nb_audios=config["dataset"]["nb_val"])
    logging.info("Build")
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    device = get_available_device()
    model = BiCodecTokenizer(device=device)
    disciminator = Discriminator()
    disciminator.to(device=device)
    optimizer_generator = torch.optim.AdamW(model.get_training_parameters_ft_bicodec(), lr=config["training"]["lr"])
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
    mlflow.set_experiment(config["training"]["experiment_name"])
    steps = 0
    with mlflow.start_run(run_name=config["training"]["run_name"] + f"_config_{config['training']['lr']}"):
        mlflow.log_params(config)
        val_metric = asdict(eval_model(dataloader=dataloader_val, model=model, mel_loss=mel_loss, device=device))  # Eval avant le début de l'entraînement
        print(val_metric)
        mlflow.log_metrics(val_metric, step=steps)
        for _ in range(config["training"]["epochs"]):
            while steps < config["training"]["nb_steps"]:
                for batch in tqdm(dataloader_train, desc="Training"):
                    batch = batch.to(device)
                    optimizer_discriminator.zero_grad()
                    with torch.no_grad():
                        outputs, _ = model(batch)
                    loss_d = compute_loss_discriminative(x=outputs, y=batch, gan_loss=gan_loss)
                    loss_d.backward()
                    nn.utils.clip_grad_norm_(disciminator.parameters(), max_norm=1.0)
                    optimizer_discriminator.step()
                    scheduler_discriminator.step()
                    optimizer_generator.zero_grad()
                    outputs, vq_metrics = model(batch)
                    loss_g = compute_loss_gen(
                            x=outputs,
                            y=batch,
                            mel_loss=mel_loss,
                            gan_loss=gan_loss,
                            apply_gan=steps>config["training"]["warmup_step_generator"]
                    )
                    loss_g += vq_metrics["vq_loss"]
                    loss_g.backward()
                    nn.utils.clip_grad_norm_(model.get_training_parameters_ft_bicodec(), max_norm=1.0)
                    optimizer_generator.step()
                    scheduler_generator.step()
                    steps += 1
                    if steps >= config["training"]["nb_steps"]:
                            break
                    if steps % config["training"]["eval_step"] == 0:
                        val_metric = asdict(eval_model(dataloader=dataloader_val, model=model, mel_loss=mel_loss, device=device))
                        mlflow.log_metrics(val_metric, step=steps)
                        path_to_save = Path(__file__).resolve().parent / "checkpoints" / f"val_loss_{val_metric['mel_loss']:.4f}_lr_{steps}"
                        state_dict = {
                            k: v for k, v in model.model.state_dict().items()
                            if'mel_transformer' not in k
                        }
                        save_file(state_dict,  Path(f"{path_to_save}.safetensors"))
        val_metric = asdict(eval_model(dataloader=dataloader_val, model=model, mel_loss=mel_loss, device=device))
        mlflow.log_metrics(val_metric, step=steps)
        path_to_save = Path(__file__).resolve().parent / "checkpoints" / f"val_loss_{val_metric['mel_loss']:.4f}_lr_{steps}"
        state_dict = {
            k: v for k, v in model.model.state_dict().items()
            if'mel_transformer' not in k
        }
        save_file(state_dict,  Path(f"{path_to_save}.safetensors"))




if __name__ == "__main__":
    current_path = Path(__file__).resolve().parent
    config_path = current_path.parent / "config" / "config.yaml"
    config = read_config(str(config_path))
    all_lr = [100]
    for lr in all_lr:
        config["training"]["epochs"] = lr
        train(config=config)