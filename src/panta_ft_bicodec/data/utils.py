""" Utils fonction mainly to deal with audio data"""
import json
import math
import random
import tarfile
from typing import Any
from pathlib import Path
from torch import Tensor
from mistral_common.audio import Audio
import numpy as np
import torch
import soundfile as sf
from torchvision import io


def read_json(path: str) -> dict[str, Any]:
    """
    Read a JSON file and return its content.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Content of the JSON file
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data: dict[str, Any]):
    if not(path.endswith(".json")):
        path += ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

def audio_preprocess(
    audio_path: str,
    sample_rate_expected: int = 16_000,
    hop_length: int | None = None,
    max_seconds: int = -1,
) -> Tensor:
    """
    Load an audio file, resample if necessary, and return a normalized 1D PyTorch tensor (mono).
    
    Args:
        audio_path: Path to the audio file
        sample_rate_expected: Target sample rate in Hz (default: 16000)
        hop_length: Optional hop length for padding alignment
        max_seconds: Maximum duration in seconds (default: 3)
        
    Returns:
        Normalized audio waveform as a 1D PyTorch tensor
    """
    waveform = Audio.from_file(audio_path, strict=False)
    
    if waveform.sampling_rate != sample_rate_expected:
        waveform.resample(sample_rate_expected)
    
    waveform = waveform.audio_array
    # Truncate if exceeds max duration
    if max_seconds > 0:
        max_length = int(max_seconds * sample_rate_expected)
        if waveform.shape[-1] > max_length:
            start_pos = random.randint(0, waveform.shape[-1] - max_length - 1)
            waveform = waveform[..., start_pos:start_pos + max_length]
    
    # Pad if hop_length is specified
    if hop_length is not None:
        length = waveform.shape[-1]
        left_pad = math.ceil(length / hop_length) * hop_length - length
        if left_pad > 0:
            # waveform : np.array de forme (N,)
            waveform = np.pad(waveform, (left_pad, 0), mode='constant', constant_values=0)
    return waveform.squeeze()


def save_signal_to_wav(
        signal: Tensor | np.ndarray,
        filename: str,
        signaling_rate: int=16_000,
        expected_sampling_rate: int=16_000
    ) -> None:
    """ Function to save audio

    Args:
        signal (Tensor | np.ndarray): 
        filename (str): name to save the audio
        signaling_rate (int, optional): Initiale sampling rate. Defaults to 16_000.
        expected_sampling_rate (int, optional): Target sampling rate. Defaults to 16_000.
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy().flatten()
    
    signal = Audio(audio_array=signal, sampling_rate=signaling_rate, format="wav")

    if signaling_rate != expected_sampling_rate:
        signal.resample(expected_sampling_rate)
    
    signal = signal.audio_array
    
    sf.write(filename, signal, expected_sampling_rate)

