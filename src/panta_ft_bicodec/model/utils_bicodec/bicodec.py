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
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
from safetensors.torch import load_file

from panta_ft_bicodec.model.utils_bicodec.file import load_config
from panta_ft_bicodec.model.utils_bicodec.modules.encoder_decoder.feat_decoder import Decoder
from panta_ft_bicodec.model.utils_bicodec.modules.encoder_decoder.feat_encoder import Encoder
from panta_ft_bicodec.model.utils_bicodec.modules.encoder_decoder.wave_generator import WaveGenerator
from panta_ft_bicodec.model.utils_bicodec.modules.speaker.speaker_encoder import SpeakerEncoder
from panta_ft_bicodec.model.utils_bicodec.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize


class BiCodec(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating a speaker encoder, feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        **kwargs
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            speaker_encoder (nn.Module): Speaker encoder module.
            prenet (nn.Module): Prenet network.
            postnet (nn.Module): Postnet network.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.init_mel_transformer(mel_params)
        # self.encoder.requires_grad_(False)
        # self.quantizer.requires_grad_(False)
        # self.speaker_encoder.requires_grad_(False)
        self.postnet.requires_grad_(False)

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.
        
        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
        )

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        model.eval()
        model.remove_weight_norm()

        return model

    def get_parameter(self):
        for param in self.prenet.parameters():
            yield param
        for param in self.decoder.parameters():
            yield param

    def get_parameter_ft_bicodec(self):
        for param in self.encoder.parameters():
            yield param
        for param in self.quantizer.parameters():
            yield param
        for param in self.prenet.parameters():
            yield param
        for param in self.decoder.parameters():
            yield param
        for param in self.speaker_encoder.parameters():
            yield param

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.
        
        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        # Extract speaker embeddings from the reference waveforms
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)
        _, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        # Encode the features and quantize them
        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)
        conditions = d_vector
        x = self.prenet(vq_outputs["z_q"], conditions)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)
        return wav_recon, vq_outputs

    @torch.no_grad()
    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)
        all_global_features = []
        all_semantic_tokens = []
        for single_features, single_mel in zip(feat, mel):
            z = self.encoder(single_features.transpose(1, 2))
            semantic_tokens = self.quantizer.tokenize(z)
            global_tokens = self.speaker_encoder.tokenize(single_mel.unsqueeze(0).transpose(1, 2))
            all_global_features.append(global_tokens.squeeze())
            all_semantic_tokens.append(semantic_tokens.squeeze())
        return all_semantic_tokens, all_global_features

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        all_wave = []
        for single_seg, single_global in zip(semantic_tokens, global_tokens):
            z_q = self.quantizer.detokenize(single_seg.unsqueeze(0))
            d_vector = self.speaker_encoder.detokenize(single_global.unsqueeze(0).unsqueeze(0))
            x = self.prenet(z_q, d_vector)
            x = x + d_vector.unsqueeze(-1)
            wav_recon = self.decoder(x)
            all_wave.append(wav_recon)
        return all_wave

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


# Test the model
if __name__ == "__main__":

    config = load_config("pretrained_models/SparkTTS-0.5B/BiCodec/config.yaml")
    model = BiCodec.load_from_checkpoint(
        model_dir="pretrained_models/SparkTTS-0.5B/BiCodec",
    )

    # Generate random inputs for testing
    duration = 0.96
    x = torch.randn(20, 1, int(duration * 16000))
    feat = torch.randn(20, int(duration * 50), 1024)
    inputs = {"feat": feat, "wav": x, "ref_wav": x}

    # Forward pass
    outputs = model(inputs)
    semantic_tokens, global_tokens = model.tokenize(inputs)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)

    # Verify if the reconstruction matches
    if torch.allclose(outputs["recons"].detach(), wav_recon):
        print("Test successful")
    else:
        print("Test failed")
