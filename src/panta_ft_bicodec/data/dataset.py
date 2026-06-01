from pathlib import Path

from torch.utils.data import Dataset
import torch

from panta_ft_bicodec.constant import MIN_SECOND, MIN_TIME_SEQUENCE, SAMPLING_RATE
from panta_ft_bicodec.model.utils_bicodec.audio import load_audio
import logging


class CustomDatasetAudio(Dataset):
    def __init__(self, list_path_audio: list | None=None) -> None:
        self.dataset_path = Path(__file__).resolve().parent / "filter_data"
        if list_path_audio is None:
            self.list_path_audio = list(self.dataset_path.iterdir())
        else:
            self.list_path_audio = []
            for folder_path in list_path_audio:
                if Path(folder_path).is_dir():
                    self.list_path_audio += list(Path(folder_path).glob("**/*.wav"))
                else:
                    self.list_path_audio += [Path(folder_path)]
        logging.info(f"Dataset initialized with {len(self.list_path_audio)} audio files.")
    
    def __len__(self) -> int:
        return len(self.list_path_audio)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        path_audio = self.list_path_audio[index]
        audio = load_audio(
            adfile=str(path_audio),
            sampling_rate=SAMPLING_RATE,
            volume_normalize=True,
            segment_duration=MIN_SECOND
        )
        return torch.from_numpy(audio).float()

    def split_dataset(self, nb_audios: int) -> "CustomDatasetAudio":
        """ Split our dataset into a sub dataset """
        if nb_audios > len(self.list_path_audio):
            raise ValueError("nb_audios is larger than the dataset size")
        sub_list_path_audio = self.list_path_audio[:nb_audios]
        self.list_path_audio = self.list_path_audio[nb_audios:]
        return CustomDatasetAudio(list_path_audio=sub_list_path_audio)


if __name__ == "__main__":
    dataset = CustomDatasetAudio()
    print(dataset[1].shape)