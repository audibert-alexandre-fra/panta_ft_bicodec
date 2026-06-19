from pathlib import Path

from tqdm import tqdm

from panta_ft_bicodec.constant import MIN_SECOND, SAMPLING_RATE
from panta_ft_bicodec.model.utils_bicodec.audio import load_audio
from panta_ft_bicodec.data.bdd import PantaDB
from dataclasses import dataclass
import shutil
import argparse



def process_remove_short_segments(bdd_path: str, name: str) -> None:
    path_to_data = Path(bdd_path).resolve().parent
    current_path = Path(__file__).resolve().parent
    dataset_path_save = current_path / name
    db = PantaDB(bdd_path)
    data = db.conn.execute("""
        SELECT name_file, duration
        FROM segments
        WHERE score_transcription < 0.02
    """).fetchall()
    durations = [d[1] for d in data]
    data_path = [d[0] for d in data]
    dataset_path_save.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    total_time = 0
    for path, duration in tqdm(zip(data_path, durations)):
        if duration > MIN_SECOND and total_time / 3600 < 300:
            shutil.copy((path_to_data/ path), dataset_path_save / Path(path).name)
            total_samples += 1
            total_time += duration
    print(f"Nb element: {total_samples}", f"Nb hours: {total_time/3600}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help'
                )
    parser.add_argument('--bdd_path', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    parse = parser.parse_args()
    process_remove_short_segments(bdd_path=parse.bdd_path, name=parse.name)