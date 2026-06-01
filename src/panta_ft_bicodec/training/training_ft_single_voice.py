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
from panta_ft_bicodec.training.loss import GANLoss, MelSpectrogramLoss
from panta_ft_bicodec.training.utils import get_available_device, set_seed
import logging
import torch.nn.functional as F
import torchaudio.transforms as T
import mlflow
from tqdm import tqdm
from safetensors.torch import load_model, save_model
from safetensors.torch import save_file

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
    set_seed(seed=config["training"]["seed"])
    dataset_train = CustomDatasetAudio()
    dataset_val = dataset_train.split_dataset(nb_audios=config["dataset"]["nb_val"])
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
    optimizer_generator = torch.optim.AdamW(model.get_training_parameters(), lr=config["training"]["lr"])
    optimizer_discriminator = torch.optim.AdamW(disciminator.parameters(), lr=config["adversarial_model"]["lr"])

    total_steps = len(dataloader_train) * config["training"]["epochs"]
    warmup_steps_generator = config["training"]["warmup_step_generator"]

    scheduler_generator = torch.optim.lr_scheduler.SequentialLR(
        optimizer_generator,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer_generator, start_factor=0.1, total_iters=warmup_steps_generator),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, T_max=total_steps - warmup_steps_generator)
        ],
        milestones=[warmup_steps_generator]
    )

    warmup_steps_disciminator = config["training"]["warmup_step_disciminator"]
    scheduler_discriminator = torch.optim.lr_scheduler.SequentialLR(
        optimizer_discriminator,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer_discriminator, start_factor=0.1, total_iters=warmup_steps_disciminator),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator, T_max=total_steps - warmup_steps_disciminator)
        ],
        milestones=[warmup_steps_disciminator]
    )
    mel_loss = MelSpectrogramLoss(weight=config["loss"]["weight_mel"])
    mel_loss.set_device(device=device)
    gan_loss = GANLoss(discriminator=disciminator)

    def compute_loss_gen(x, y, apply_gan=True):
        min_len = min(x.shape[-1], y.shape[-1])
        x = x[..., :min_len]
        y = y[..., :min_len]
        loss = 0.0
        loss += mel_loss(x, y)
        if apply_gan:
            loss_g, loss_feature = gan_loss.generator_loss(fake=x, real=y)
            loss += loss_g + loss_feature
        return loss
    
    def compute_loss_discriminative(x, y):
        return gan_loss.discriminator_loss(fake=x, real=y)

    mlflow.set_experiment(config["training"]["experiment_name"])
    step = 0
    with mlflow.start_run(run_name=config["training"]["run_name"] + f"_config_{config['training']['lr']}"):
            mlflow.log_params(config)
            def eval_model(step, model, dataloader, device):
                print("Start eval")
                model.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in dataloader:
                        batch = batch.to(device)
                        outputs = model(batch)
                        loss = compute_loss_gen(x=outputs, y=batch, apply_gan=False)
                        val_loss += loss.item()
                val_loss /= len(dataloader)
                logging.info(f"Step {step} | Val loss: {val_loss:.4f}")
                mlflow.log_metric("val_loss", val_loss, step=step)
                model.model.train()
                return val_loss
            eval_model(step=step, model=model, dataloader=dataloader_val, device=device)  # Eval avant le début de l'entraînement
            for epoch in range(config["training"]["epochs"]):
                logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
                print("start training")
                for batch in tqdm(dataloader_train, desc="Training"):
                    batch = batch.to(device)
                    optimizer_discriminator.zero_grad()
                    with torch.no_grad():
                        outputs = model(batch)
                    loss_d = compute_loss_discriminative(x=outputs, y=batch)
                    loss_d.backward()
                    nn.utils.clip_grad_norm_(disciminator.parameters(), max_norm=1.0)
                    optimizer_discriminator.step()
                    scheduler_discriminator.step()
                    optimizer_generator.zero_grad()
                    outputs = model(batch)
                    loss_g = compute_loss_gen(x=outputs, y=batch, apply_gan=step>warmup_steps_generator)
                    loss_g.backward()
                    nn.utils.clip_grad_norm_(model.get_training_parameters(), max_norm=1.0)
                    optimizer_generator.step()
                    scheduler_generator.step()
                    step += 1
                # VAL
                val_loss = eval_model(step=step, model=model, dataloader=dataloader_val, device=device)  # Eval avant le début de l'entraînement
                path_to_save = Path(__file__).resolve().parent / "checkpoints" / f"val_loss_{val_loss:.4f}_lr_{epoch}"

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