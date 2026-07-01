import os
from pathlib import Path

from transformers.audio_utils import load_audio

from panta_ft_bicodec.constant import SAMPLING_RATE
from panta_ft_bicodec.evaluate.utils import save_audio
from panta_ft_bicodec.model.bicodec_tokenizer import BiCodecTokenizer
from panta_ft_bicodec.training.utils import get_available_device
import logging
import argparse

logging.basicConfig(level=logging.INFO)

def build_audio_baseline(path_to_model: str):
    print("IN")
    folder_to_save = "test"
    logging.info(f" model name {path_to_model}")
    device = "cpu" #get_available_device()
    logging.info(f" Device used during evaluation: {device}")
    model = BiCodecTokenizer(device=device)
    model.load_trained_model(path_to_model=path_to_model)
    os.makedirs(folder_to_save, exist_ok=True)
    current_dir = Path("ref") #Path("/home/getalp/audibeal/build_dataset_audio/repere")
    for index, file in enumerate(current_dir.glob("*.wav")):
        gobal_tokens, sementic_tokens = model.tokenize(str(file))
        audio_reconstructed = model.detokenize(gobal_tokens, sementic_tokens)
        save_audio(audio_reconstructed, f"{folder_to_save}/{file.stem}.wav", SAMPLING_RATE)
        # if index > 10:
        #     break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name_model", type=str)
    args = parser.parse_args()
    path_to_model = Path(__file__).parent.parent / "training" / "checkpoints" / args.name_model
    build_audio_baseline(path_to_model=path_to_model)
