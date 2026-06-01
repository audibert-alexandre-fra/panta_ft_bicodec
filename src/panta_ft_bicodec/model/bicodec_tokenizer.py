# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np

from pathlib import Path
from typing import Any, Dict, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from panta_ft_bicodec.model.utils_bicodec.audio import load_audio
from panta_ft_bicodec.model.utils_bicodec.bicodec import BiCodec
from panta_ft_bicodec.model.utils_bicodec.file import load_config
from safetensors.torch  import load_file, load_model


class BiCodecTokenizer:
    """BiCodec tokenizer for handling audio input and tokenization."""

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__()
        """
        Args:
            model_dir: Path to the model directory.
            device: Device to run the model on (default is GPU if available).
        """
        self.path_to_config = Path(__file__).resolve().parent.parent / "pre_trained_weights" / "bicodec"
        self.device = device
        self.config = load_config( self.path_to_config / "config.yaml")
        self._initialize_model()

    def _initialize_model(self):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        self.model = BiCodec.load_from_checkpoint(self.path_to_config / "BiCodec").to(
            self.device
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.path_to_config  / "wav2vec2-large-xlsr-53"
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            self.path_to_config  / "wav2vec2-large-xlsr-53",
            use_safetensors=False
        ).to(self.device)
        self.feature_extractor.eval()
        self.feature_extractor.config.output_hidden_states = True
        self.feature_extractor.requires_grad_(False)
    
    def load_trained_model(self, path_to_model):
        state_dict = load_file(path_to_model)
        state_dict = {k: v for k, v in state_dict.items() if "window" not in k}
        self.model.load_state_dict(state_dict, strict=False)


    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = (
            int(self.config["sample_rate"] * 5)#self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def process_audio(self, wav_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """load auido and get reference audio from wav path"""
        wav = load_audio(
            wav_path,
            sampling_rate=self.config["sample_rate"],
            volume_normalize=self.config["volume_normalize"],
        )

        wav_ref = self.get_ref_clip(wav)

        wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float()
        return wav, wav_ref


    @torch.no_grad()
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        all_attention = []
        all_features = []
        if len(wavs.shape) == 1:
            wavs = wavs.reshape(1, -1)
        for single_audio in wavs:
            inputs = self.processor(
                [single_audio.cpu().squeeze() if isinstance(single_audio, torch.Tensor) else single_audio],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                output_hidden_states=True,
            )

            attention_mask = inputs.attention_mask
            input_values = inputs.input_values.squeeze(1)
            feat = self.feature_extractor(
                input_values.to(self.feature_extractor.device),
                attention_mask=attention_mask.to(self.feature_extractor.device),
            )
            feats_mix = (
                feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
            ) / 3
            all_attention.append(attention_mask)
            all_features.append(feats_mix)
        return all_features, all_attention

    @torch.no_grad()
    def tokenize_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """tokenize the batch of audio

        Args:
            batch:
                wavs (List[np.ndarray]): batch of audio
                ref_wavs (torch.Tensor): reference audio. shape: (batch_size, seq_len)

        Returns:
            semantic_tokens: semantic tokens. shape: (batch_size, seq_len, latent_dim)
            global_tokens: global tokens. shape: (batch_size, seq_len, global_dim)
        """
        feats, attention_mask = self.extract_wav2vec2_features(batch["wav"])
        batch["feat"] = feats
        batch["ref_wav"] = batch["ref_wav"].to(self.device)
        semantic_tokens, global_tokens = self.model.tokenize(batch)
        return global_tokens, semantic_tokens, attention_mask

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio"""
        wav, ref_wav = self.process_audio(audio_path)
        feat, _ = self.extract_wav2vec2_features(wav)
        batch = {
            "wav": torch.from_numpy(wav).unsqueeze(0).float().to(self.device),
            "ref_wav": ref_wav.to(self.device),
            "feat": torch.stack(feat),
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.ndarray:
        """detokenize the tokens to waveform

        Args:
            global_tokens: global tokens. shape: (batch_size, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        return [wav.detach().squeeze().cpu().numpy() for wav in wav_rec]
    
    def get_training_parameters(self):
        """get parameters for training"""
        return self.model.get_parameter()

    def get_training_parameters_ft_bicodec(self):
        """get parameters for training"""
        return self.model.get_parameter_ft_bicodec()

    def __call__(self, batch: torch.Tensor):
        feats, _ = self.extract_wav2vec2_features(batch)
        inputs_data = {}
        inputs_data["feat"] = torch.stack(feats).squeeze(1)
        inputs_data["ref_wav"] = batch.to(self.device)
        return self.model(inputs_data)



if __name__ == "__main__":
    tokenizer = BiCodecTokenizer()
