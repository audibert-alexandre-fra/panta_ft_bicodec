import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
import torchaudio.transforms as T
from safetensors.torch import save_file, load_model
from panta_ft_bicodec.constant import SAMPLING_RATE


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []
        x = self.pad_to_period(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = SAMPLING_RATE,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.n_fft = window_length // 2 + 1
        self.stft_params = torch.hann_window(self.window_length )
        bands = [(int(b[0] * self.n_fft), int(b[1] * self.n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)  # (B, 1, T) → (B, T)
        
        x_stft = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=int(self.window_length * self.hop_factor),
            window=self.stft_params.to(x.device),
            return_complex=True
        ) # → (B, freq, time)
        
        x_real = torch.view_as_real(x_stft)      # (B, freq, time, 2)
        x_real = x_real.permute(0, 3, 2, 1)      # (B, 2, time, freq)
        x_bands = [x_real[..., b[0]:b[1]] for b in self.bands]

        fmap = []
        x_out = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x_out.append(band)
        
        x_out = torch.cat(x_out, dim=-1)
        x_out = self.conv_post(x_out)
        fmap.append(x_out)
        return fmap


class Discriminator(nn.Module):
    def __init__(
        self,
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = SAMPLING_RATE,
        bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p) for p in periods]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps

    def save(self, path):
        """Sauvegarde les poids du discriminator (state_dict natif, sans préfixe)."""
        save_file(self.state_dict(), str(path))

    def load(self, path, device: str = "cpu", strict: bool = True):
        """Charge les poids du discriminator depuis un fichier .safetensors."""
        load_model(self, str(path), strict=strict)
    


if __name__ == "__main__":
    disc = Discriminator()
    disc.save(path="test.safetensors")
    disc.load(path="test.safetensors")
    # x = torch.zeros(1, 1, int(SAMPLING_RATE * 4.99))
    # results = disc(x)
    # for i, result in enumerate(results):
    #     print(f"disc{i}")
    #     for i, r in enumerate(result):
    #         print(r.shape, r.mean(), r.min(), r.max())
    #     print()