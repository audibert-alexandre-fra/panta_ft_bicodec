from mistral_common.audio import Audio
from pathlib import Path


total_seconde = 0
path_to_data = Path(__file__).resolve().parent / "filter_data" 
for path in path_to_data.iterdir():
    audio = Audio.from_file(path, strict=False)
    total_seconde += audio.duration

print(f"total seconde: {total_seconde/3600:.2f}h")
