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
    current_path = current_dir = Path(__file__).resolve().parent
    device = get_available_device()
    logging.info(f" Device used during evaluation: {device}")
    model = BiCodecTokenizer(device=device) 
    os.makedirs("audio_baseline", exist_ok=True)
    current_dir = current_path.parent / "data" / "test_data"
    os.makedirs('ref', exist_ok=True)
    for index, file in enumerate(current_dir.glob("*.wav")):
        shutil.copy(str(file), f'ref/{file.stem}.wav')
        gobal_tokens, sementic_tokens = model.tokenize(str(file))
        audio_reconstructed = model.detokenize(gobal_tokens, sementic_tokens)
        save_audio(audio_reconstructed, f"audio_baseline/{file.stem}.wav", SAMPLING_RATE)


if __name__ == '__main__':
    build_audio_baseline()
