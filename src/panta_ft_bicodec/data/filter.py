from pathlib import Path

from tqdm import tqdm

from panta_ft_bicodec.constant import MIN_SECOND, SAMPLING_RATE
from panta_ft_bicodec.model.utils_bicodec.audio import load_audio
import shutil
import argparse

def process_remove_short_segments(dataset_path: str, name: str) -> None:
    current_path = Path(__file__).resolve().parent
    dataset_path = Path(dataset_path).resolve()
    dataset_path_save = current_path / name
    files = []
    files = []
    files += list(dataset_path.rglob("*.wav"))
    files += list(dataset_path.rglob("*.mp3"))
    files += list(dataset_path.rglob("*.mp4"))
    dataset_path_save.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    total_time = 0
    for wav in files:
        audio = load_audio(
            adfile=str(wav),
            sampling_rate=SAMPLING_RATE,
        )
        if len(audio) > MIN_SECOND * SAMPLING_RATE:
            shutil.copy(wav, dataset_path_save / wav.name)
            total_samples += 1
            total_time += len(audio) / SAMPLING_RATE
    print(f"Nb element: {total_samples}", f"Nb hours: {total_time/3600}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help'
                )
    parser.add_argument('--data', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    parse = parser.parse_args()
    process_remove_short_segments(dataset_path=parse.data, name=parse.name)