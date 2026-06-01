from panta_ft_bicodec.constant import SAMPLING_RATE
import soundfile as sf
import numpy as np


def save_audio( audio_array: np.ndarray, path_to_save: str, sampling_rate: int=SAMPLING_RATE):
    sf.write(path_to_save, audio_array[0], samplerate=sampling_rate)