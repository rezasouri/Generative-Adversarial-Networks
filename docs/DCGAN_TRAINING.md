# DCGAN baseline training

ForgeGAN now includes an **unconditional** 64×64 RGB DCGAN training baseline. This stage does not synthesize defect masks or condition generation on defect type: the generator learns the distribution of **normal training images** only.

## Install and run

Install PyTorch and training dependencies with:

```bash
python -m pip install -e ".[train,dev]"
forgegan train-dcgan --config configs/experiments/mvtec-bottle-dcgan.yaml
```

Obtain and extract the dataset separately through its official distributor. The example config expects `datasets/mvtec_ad/bottle/train/good`. For VisA, set `dataset.name: visa`, its extracted root and the official CSV path. Set `dataset.split: train`; anomalous samples are intentionally excluded.

To resume from the latest completed epoch:

```bash
forgegan train-dcgan \
  --config configs/experiments/mvtec-bottle-dcgan.yaml \
  --resume runs/mvtec-bottle-dcgan-baseline/checkpoint-last.pt
```

The `--device` option supports `auto`, `cpu` or `cuda`. The model has fixed 64×64 RGB output, and dataset images are resized to 64×64. Generator output uses Tanh in [-1, 1]; real images are mapped from [0, 1] to [-1, 1]. Discriminator uses logits and BCEWithLogitsLoss. Training alternates discriminator and generator optimization with Adam.

## Checkpoint and samples

The output directory contains:

- `experiment.json` — configuration and resume identity
- `losses.jsonl` — one row per completed epoch with average generator and discriminator losses
- `samples/epoch-NNNN.png` — grid from fixed noise for visual comparison
- `checkpoint-last.pt` — generator, discriminator, optimizer states, RNG and DataLoader generator state

Checkpoint writes use a temporary file and replace operation. Resuming requires the same model hyperparameters, dataset definition, seed and training settings. `training.epochs` is the **total target number of epochs**, not additional epochs. Resume restarts at the next epoch; checkpoints within an incomplete epoch are not supported.

Load checkpoints only from trusted sources. PyTorch loading uses `weights_only=True` to restrict deserialization. Existing output logs are protected against accidental overwrite on a fresh run. For testing, set `training.max_batches_per_epoch: 1` to train only one minibatch per epoch; report this limitation in any outputs.

## Benchmark caveats

This baseline tests generative training plumbing. A visually plausible sample grid is not proof of useful defect synthesis, nor an industrial anomaly-detection benchmark. For meaningful results, report FID/KID or task-appropriate image-quality measures, multiple random seeds, and the downstream real-only versus augmented anomaly-detection experiment with a fixed, untouched test split.

## Testing

The PyTorch CI job exercises Generator/Discriminator shapes, a small synthetic fixture through one epoch, checkpoint persistence, second-epoch resume, and official test-split rejection. The lightweight quality job runs non-PyTorch tests and skips optional training tests.

The integration fixture is synthetic and does not replace training on the actual MVTec AD or VisA dataset.
