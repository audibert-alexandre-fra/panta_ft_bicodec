from pathlib import Path

from tqdm import tqdm

from panta_ft_bicodec.constant import MIN_SECOND
from panta_ft_bicodec.model.utils_bicodec.audio import load_audio
import shutil


def process_remove_short_segments():
    current_path = Path(__file__).resolve().parent
    path_to_wav = current_path / "data_ft"
    for wav in tqdm(path_to_wav.iterdir()):
        audio = load_audio(
            adfile=str(wav),
            sampling_rate=16000,
            )
        if len(audio) > MIN_SECOND * 16000:
            shutil.copy(wav, current_path / "filter_data" / wav.name)


if __name__ == "__main__":
    process_remove_short_segments()