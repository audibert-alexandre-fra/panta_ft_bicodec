from dataclasses import dataclass
from typing import List
import typing

from torch import Tensor, nn
import torchaudio.transforms as T
import torch

from panta_ft_bicodec.constant import SAMPLING_RATE 
import torch.nn.functional as F


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    weight : float, optional
        Weight of this loss, by default 1.0

    """

    def __init__(
        self,
        n_mels: int = 80,
        window_lengths: List[int] = [512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        weight: float = 1.0,
    ):
        super().__init__()
        self.mel_constructor = nn.ModuleList([
            T.MelSpectrogram(
                sample_rate=SAMPLING_RATE,
                n_fft=w,
                hop_length=w//4,
                n_mels=n_mels
            )
            for w in window_lengths
        ])
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.weight = weight

    def forward(self, x: Tensor, y: Tensor):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for mel_spectogramm_transform in self.mel_constructor:
            x_mels = mel_spectogramm_transform(x).squeeze(1)
            y_mels = mel_spectogramm_transform(y).squeeze(1)

            loss += self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(2.0).log10(),
                y_mels.clamp(self.clamp_eps).pow(2.0).log10(),
            )
            loss += self.loss_fn(x_mels, y_mels)
        return loss * self.weight


class L1LossWeighted(nn.L1Loss):

    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.weight * super().forward(x, y)


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake: Tensor, real: Tensor):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature




def compute_loss_gen(x, y, mel_loss, gan_loss, apply_gan=True):
    min_len = min(x.shape[-1], y.shape[-1])
    x = x[..., :min_len]
    y = y[..., :min_len]
    loss = 0.0
    loss += mel_loss(x, y)
    if apply_gan:
        loss_g, loss_feature = gan_loss.generator_loss(fake=x, real=y)
        loss += loss_g + loss_feature
    return loss


def compute_loss_discriminative(x, y, gan_loss):
    return gan_loss.discriminator_loss(fake=x, real=y)


@dataclass
class ValidationOutput:
    mel_loss: float
    vq_usage: float
    loss_vq: float
    discriminator_loss: float



@torch.inference_mode
def eval_model(dataloader, model, mel_loss, gan_loss, device) -> ValidationOutput:
    model.model.eval()
    total_mel_loss = 0
    loss_vq = 0
    vq_usage = []
    loss_disciminator = 0
    for batch in dataloader:
        batch = batch.to(device)
        outputs, vq_metric = model(batch)
        total_mel_loss += compute_loss_gen(outputs, batch, mel_loss=mel_loss, gan_loss=None, apply_gan=False).item()
        vq_usage.append(vq_metric["indices"])
        loss_disciminator += gan_loss.discriminator_loss(outputs, batch)
        loss_vq += vq_metric["vq_loss"].item()
    model.model.train()
    return ValidationOutput(
        mel_loss=total_mel_loss / len(dataloader),
        vq_usage=torch.cat(vq_usage).flatten().unique().shape[0]/8196,
        discriminator_loss=loss_disciminator / len(dataloader),
        loss_vq=loss_vq / len(dataloader)
    )



