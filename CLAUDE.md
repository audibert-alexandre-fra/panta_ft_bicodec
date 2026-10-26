# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tuning of the **BiCodec** speech codec (from SparkAudio / Spark-TTS) with a GAN setup (mel-spectrogram reconstruction loss + multi-period / multi-resolution discriminator) on French-ish speech corpora (MLS, EPAC). Training runs on the **Jean Zay** supercomputer (SLURM, H100 nodes, `idr_torch`). There are no tests and no linter configured; the root `main.py` is a placeholder.

## Environment

- Python 3.11, managed with `uv` (`uv.lock`, PyTorch from the cu128 index). Install with `uv sync`; scripts assume `.venv/` at the repo root.
- The package is `src/panta_ft_bicodec` and all imports are absolute (`from panta_ft_bicodec....`), so it must be installed in the venv (done by `uv sync`).

## The two training scripts

Both live in `src/panta_ft_bicodec/training/` and share the same GAN loop, losses, checkpoint format and config schema:

| | `training_ft_bicodec.py` | `training_ft_single_voice.py` |
|---|---|---|
| Purpose | Fine-tune the **whole** BiCodec | Fine-tune the **decoder only** (e.g. adapt to one voice) |
| Trained params | `get_parameter_ft_bicodec()`: encoder, quantizer, prenet, decoder, speaker_encoder | `get_parameter()`: prenet + decoder; encoder, quantizer, speaker_encoder, postnet are explicitly frozen |
| Starting weights | Pretrained BiCodec, or resume via `training.load_model` | **Hardcoded** full-FT checkpoint `lr_2e-05_lr_ad3e-05_norm_disc_5_norm_gen_10/val_loss_51_lr_20` (`load_model` is ignored, optimizers start fresh) |
| Generator loss | mel + GAN ramped in over steps 20k→30k + VQ loss | mel + GAN from step 0, no VQ loss (quantizer frozen) |
| Output dir | `lr_<lr>_lr_ad<lr_ad>_norm_disc_<n>_norm_gen_<n>/` | `ft_lr_<lr>_steps_<nb_steps>/` |
| Typical launch | `launch_N.slurm` + `config_N.yaml` | `launch_continue.slurm` + `config_1_ft.yaml` |

**Freezing mechanism:** `BiCodec.__init__` freezes encoder, quantizer and speaker_encoder by default, and `BiCodec.forward` only builds a graph through them when `finetune_encoder` is set. So the two scripts above both train only prenet + decoder. The newer `training_full.py` (everything trained) and `training_decoder.py` (prenet + decoder) share an identical loop. Each picks its trained modules in `select_trainable_modules()`, and `diff` between the two shows exactly how they differ. The ECAPA x-vector branch (`pool`, `bn`, `linear`, `bn2`) always stays frozen because it does not feed `d_vector`; making it trainable makes DDP fail on parameters that get no gradient. With the decoder-only script, the ECAPA BatchNorm running stats still drift, since the model is in train mode.

A change to the shared logic (loop, losses, saving) usually has to be applied to both files.

## Running

All entry points are run **from their own directory**, because paths are resolved relative to `__file__` or to the CWD:

```bash
# Multi-node training on Jean Zay (from src/panta_ft_bicodec/training/)
sbatch launch_8.slurm            # -> srun python -u training_ft_bicodec.py --config config_8.yaml

# Fine-tuning a single voice from an already fine-tuned checkpoint
sbatch launch_continue.slurm     # -> training_ft_single_voice.py --config config_1_ft.yaml

# Local multi-GPU run (set training.type_ddp: "single_node" in the config)
torchrun --nproc_per_node=<N> training_ft_bicodec.py --config config_X.yaml

# Evaluation (from src/panta_ft_bicodec/evaluate/): reconstruct every wav in ./ref/
python3 run_model.py <checkpoint>.safetensors   # looks in training/checkpoints/
python3 run_baseline.py                          # original BiCodec, no fine-tuning
```

- `--config` is a **file name**, resolved to `src/panta_ft_bicodec/config/<name>`. Each `launch_N.slurm` pairs with `config_N.yaml`; SLURM logs go to `training/logs_slurm/`.
- `train.sh` at the root references a non-existent `training.py` and is stale.
- Training scripts always call `dist.init_process_group` — there is no non-distributed path. `type_ddp` must be `"jeanzay"` (uses `idr_torch` env) or `"single_node"` (uses torchrun env vars).

## Architecture

- **`model/bicodec_tokenizer.py` – `BiCodecTokenizer`**: wrapper (not an `nn.Module`) that holds `self.model` (the trainable `BiCodec`) plus a frozen `wav2vec2-large-xlsr-53` feature extractor (mean of hidden layers 11/14/16 as semantic features). `__call__(batch_wav)` runs the full encode→quantize→decode pass used in training; `tokenize`/`detokenize` are for inference. Pretrained weights/configs are loaded from `pre_trained_weights/bicodec/` (weights are gitignored and must be present locally).
- **`model/utils_bicodec/`**: vendored SparkAudio BiCodec code (encoder, FSQ/factorized VQ, speaker encoder, Vocos-based decoder). `bicodec.py` defines the two trainable-parameter sets used by the two training scripts (see above).
- **`model/disciminator.py`**: MPD + MRD discriminator with its own `save`/`load` (safetensors).
- **`training/loss.py`**: multi-scale `MelSpectrogramLoss`, `GANLoss` (adversarial + feature matching), `eval_model` returning a `ValidationOutput` dataclass logged to MLflow.
- **Training loop** (shared by both scripts): step-based (`nb_steps`), discriminator update then generator update each step, separate AdamW + `CosineSchedulerWithInternalState` (timm cosine with warmup, stepped every iteration) per network. The GAN term in the generator loss is ramped in linearly between steps 20k and 30k (**hardcoded**, independent of the `warmup_step_*` config keys), and the VQ loss is added.
- **Checkpoints**: after each epoch (rank 0 only) into `training/<run_dir>/` where `run_dir` is built from lr / adversarial lr / grad-norm values (`build_folder_to_save`). Each checkpoint is a triple sharing a prefix `val_loss_<mel>_lr_<epoch>`: `.safetensors` (generator, `mel_transformer` keys excluded), `_disc.safetensors`, `.pt` (optimizers, schedulers, steps, epoch). Resume by setting `training.load_model` to that prefix (no extension) — it is looked up inside the same `run_dir`, so the hyperparameters that name the folder must be unchanged.
- **MLflow** tracks to a local `mlruns/` in the CWD (experiment/run name from config).
- In `training/utils_multi_gpu.py`, `save_optimizers_and_steps` is defined twice; the second definition (generator/discriminator signature) is the one in effect.

## Data

- `data/dataset.py` – `CustomDatasetAudio` globs `**/*.wav` from `dataset.list_path_audio` (absolute Lustre paths in configs), loads at 16 kHz with volume normalization and crops to `MIN_SECOND` (6 s, `constant.py`). The validation set is the first `nb_val` files (`split_dataset`, not shuffled).
- Preprocessing produces those folders: `data/filter.py --data <corpus_dir> --name <out>` keeps segments longer than 6 s (run via `pull_data*.slurm` on the `archive` partition); `filter_bdd.py` does the same from a SQLite `PantaDB` (`bdd.py`; `bbd.py` is an older duplicate) filtering on transcription score, capped at 300 h.
- Generated data, checkpoints, `mlruns`, `*.wav`, `*.safetensors`, SLURM `*.out/*.err` are gitignored.
