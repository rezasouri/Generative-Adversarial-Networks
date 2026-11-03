# Category-conditional GAN

This milestone adds a **category-conditioned** GAN (`model.name: cgan`) for normal industrial training images. It can synthesize different *object categories*, not different defect types. This distinction is important: the official one-class training splits of MVTec AD and VisA contain only normal images, so their anomalous **test** images must not be used for GAN training.

## Architecture

- ConditionalGenerator: maps latent noise and a learned category embedding to 64×64 RGB images, scaled to [-1, 1].
- ConditionalDiscriminator: combines an image with a tiled category embedding and emits a real/fake logit.
- Category IDs: derived from sorted training categories, saved in checkpoints and checked on resume.
- Uses Adam, BCEWithLogitsLoss, and the same fixed-noise / per-epoch checkpoint workflow as DCGAN.

## Training

Install the optional train dependencies and extract the dataset from its official source:

```bash
python -m pip install -e ".[train,dev]"
forgegan train-cgan --config configs/experiments/mvtec-all-categories-cgan.yaml
```

MVTec AD requires all 15 category directories when `dataset.category: null`. VisA can be used analogously with the official `split_csv/1cls.csv`; it requires all 12 category entries and their training images. The trainer intentionally refuses a dataset restricted to a single category because that would not be a meaningful category-conditional experiment.

To resume, increase `training.epochs` to the new **total** number of epochs in the YAML and run:

```bash
forgegan train-cgan \
  --config configs/experiments/mvtec-all-categories-cgan.yaml \
  --resume runs/mvtec-all-categories-cgan/checkpoint-last.pt
```

The output contains a `checkpoint-last.pt` with category vocabulary, model/optimizer states and RNG state, one sample grid per completed epoch, `experiment.json`, and per-epoch `losses.jsonl`.

## Safety and benchmark integrity

Do not train on `test` or on anomaly masks from the official test set; the trainer rejects non-train splits and non-normal labels. Category conditioning is **not** mask conditioning or controlled defect synthesis. A future defect-conditioned trainer needs a separately licensed, training-eligible dataset of paired normal/defective images or an explicit synthetic-mask training protocol that does not contaminate held-out benchmarks. This milestone does not report image quality or downstream anomaly-detection improvements.

CI runs miniature synthetic fixtures for the forward pass, out-of-range condition rejection, normal training, checkpointing and resuming. CI does not train on original industrial images.
