import os
from pathlib import Path

from transformers.audio_utils import load_audio

from panta_ft_bicodec.constant import SAMPLING_RATE
from panta_ft_bicodec.evaluate.utils import save_audio
from panta_ft_bicodec.model.bicodec_tokenizer import BiCodecTokenizer
from panta_ft_bicodec.training.utils import get_available_device
import logging
import shutil

logging.basicConfig(level=logging.INFO)

def build_audio_baseline():
    folder_dir_output = "repere_bicodec"
    device = get_available_device()
    logging.info(f" Device used during evaluation: {device}")
    model = BiCodecTokenizer(device=device) 
    os.makedirs(folder_dir_output, exist_ok=True)
    current_dir = Path("/home/getalp/audibeal/build_dataset_audio/repere")
    for index, file in enumerate(current_dir.glob("*.wav")):
        print(file)
        gobal_tokens, sementic_tokens = model.tokenize(str(file))
        audio_reconstructed = model.detokenize(gobal_tokens, sementic_tokens)
        save_audio(audio_reconstructed, f"{folder_dir_output}/{file.stem}.wav", SAMPLING_RATE)


if __name__ == '__main__':
    build_audio_baseline()
