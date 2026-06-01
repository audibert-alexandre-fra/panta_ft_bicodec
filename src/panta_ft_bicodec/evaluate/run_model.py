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
    logging.info(f" model name {path_to_model}")
    current_path = current_dir = Path(__file__).resolve().parent
    device = get_available_device()
    logging.info(f" Device used during evaluation: {device}")
    model = BiCodecTokenizer(device=device)
    model.load_trained_model(path_to_model=path_to_model)
    os.makedirs("audio_trained", exist_ok=True)
    current_dir = current_path.parent / "data" / "test_data"
    for index, file in enumerate(current_dir.glob("*.wav")):
        gobal_tokens, sementic_tokens = model.tokenize(str(file))
        audio_reconstructed = model.detokenize(gobal_tokens, sementic_tokens)
        save_audio(audio_reconstructed, f"audio_trained/{file.stem}.wav", SAMPLING_RATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name_model", type=str)
    args = parser.parse_args()
    path_to_model = Path(__file__).parent.parent / "training" / "checkpoints" / args.name_model
    build_audio_baseline(path_to_model=path_to_model)
