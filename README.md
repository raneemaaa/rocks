# LITHOS self-supervised pretraining

A practice run before applying self-supervised pretraining to a company dataset:
pretrain several self-supervised architectures on the public
[LITHOS dataset](https://www.kaggle.com/datasets/paolaruizpuentes/lithos-dataset)
(petrographic thin-section images, plane-polarized light only), compare them with a
standard evaluation protocol, and keep the best checkpoint of each for later
fine-tuning.

## What's in here, and why

| Method | File | The one-sentence idea |
|---|---|---|
| **DINO** (v1) | [`src/lithos_ssl/models/dino.py`](src/lithos_ssl/models/dino.py) | A student network matches a momentum-averaged teacher's output on different crops of the same image -- no labels, no negative pairs. |
| **DINOv2** | [`src/lithos_ssl/models/dinov2.py`](src/lithos_ssl/models/dinov2.py) | DINO, plus a masked-patch-prediction loss (iBOT) for finer-grained features and a KoLeo regularizer that helps nearest-neighbour-style evaluation. |
| **MSN** | [`src/lithos_ssl/models/msn.py`](src/lithos_ssl/models/msn.py) | DINO-like momentum setup, but the student's views are heavily patch-masked, and both views are compared via soft assignment to a learned prototype bank instead of directly. |
| **SimCLR** | [`src/lithos_ssl/models/simclr.py`](src/lithos_ssl/models/simclr.py) | The simple baseline: two augmented views of an image are pulled together, every other image in the batch is pushed apart (a contrastive loss). Cheapest to train, useful as a sanity-check floor. |
| **DINOv3** (frozen) | [`src/lithos_ssl/models/frozen_backbone.py`](src/lithos_ssl/models/frozen_backbone.py) | Not trained here at all -- [Lightly doesn't implement DINOv3](https://github.com/lightly-ai/lightly), and nobody pretrains a model at that scale from scratch on ~200k domain images anyway. We just load Meta's pretrained weights and measure how good its off-the-shelf features already are, as a reference point for the four methods above. |

Every trainable method's file has a longer docstring at the top explaining the
mechanism in plain English -- start there if you want the intuition, not just the code.

All four trainable methods are implemented by closely following
[Lightly](https://github.com/lightly-ai/lightly)'s own official
`examples/pytorch_lightning/` reference implementations (read directly from the
installed `lightly==1.5.26` source and from Lightly's GitHub repo while building this),
restructured to: read hyperparameters from [Hydra](https://hydra.cc) configs instead of
being hardcoded, expose a uniform `.embed()` method for evaluation, and log metrics
PyTorch Lightning can checkpoint on.

### Project layout

```
configs/            Hydra configs -- one YAML per data/model/eval choice
src/lithos_ssl/
  data/              dataset, PPL/XPL filtering, section-level splitting, DataModule
  models/            one LightningModule per method, + the frozen DINOv3 baseline
  eval/              embeddings -> kNN accuracy / linear-probe accuracy, + the
                     "online kNN eval" callback that drives checkpoint selection
  utils/             small shared helpers (seeding)
scripts/
  prepare_data.py           join the real download's raw files into one clean index CSV
  make_synthetic_dataset.py generate a tiny fake dataset for a fast local smoke test
  train.py                  train one method (Hydra entrypoint)
  evaluate.py                linear-probe + kNN evaluation of one checkpoint
  aggregate_results.py       rebuild results/RESULTS.md from results/runs/*.json
tests/               pytest suite against synthetic data -- no network, no real dataset
results/             RESULTS.md (auto-generated table) + per-run JSON + TensorBoard logs
checkpoints/         <method>/best.ckpt and last.ckpt per run (gitignored, see below)
```

## Only PPL, guaranteed

You asked for plane-polarized light (PPL) only, never cross-polarized (XPL). LITHOS
pairs every PPL patch with an XPL patch of the same spot (the dataset authors' own
loader finds the XPL partner of a PPL path with a literal `"ppl-0" -> "xpl-0"` filename
substitution). This project never performs that substitution, never reads an
XPL-related column, and has three independent, redundant guards against an XPL path
ever being loaded:

1. [`scripts/prepare_data.py`](scripts/prepare_data.py) only reports/validates PPL rows.
2. [`LithosDataModule.setup()`](src/lithos_ssl/data/datamodule.py) filters the raw CSV
   down to PPL-marked paths before anything else happens.
3. [`LithosPPLDataset.__init__`](src/lithos_ssl/data/lithos_dataset.py) raises an error
   if it's ever handed a non-PPL path, as a last-resort safety net.

There is no config flag that turns XPL on.

## Getting the data (you need to do this step)

The Kaggle dataset page requires a logged-in account to even browse its file listing,
so entering credentials to get past that login was always going to be your step, not
something done automatically on your behalf. Once your `kaggle.json` API token was in
place, though, that same login let us authenticate the Kaggle CLI and actually inspect
the real file layout (`kaggle datasets files`, plus downloading the two small
fold-index CSVs and one sample patch) -- so the schema this project is built against is
now **confirmed against the real dataset**, not just inferred from the
[dataset authors' public code](https://github.com/BCV-Uniandes/LITHOS/blob/main/data_loader.py)
the way it was initially. Specifically, confirmed: the real layout is *two-level*, not
one flat CSV --

- `Fold1_complete_info_allclasses.csv` / `Fold2_complete_info_allclasses.csv` at the
  dataset root, each listing PPL patch paths (`Sec_patch`) for one of two
  cross-validation folds. Together they list all 105,802 labeled patches -- which
  matches the dataset's documented "105,802 expert-annotated grains" figure exactly
  (the documented "211,604 patches" figure is these same locations counted twice, once
  per polarization).
- every patch has its own folder (e.g. `GMAS_LOTE/7/132/`) containing `ppl-0.png`,
  `xpl-0.png`, and its *own* small `center_component.csv` with that patch's
  `Multiclass_label`/`Binary_label`.
- the thin-section id is two directory levels above the image file (e.g.
  `GMAS_LOTE/7`) -- confirmed to give exactly 580 distinct values across the real
  105,802-row index, matching the documented "580 thin sections" exactly.

`scripts/prepare_data.py` does the one-time join of those two levels into a single
`patch_index.csv` (`Sec_patch`/`Binary_label`/`Multiclass_label`), which is the only
file the rest of this codebase actually reads -- see the module docstring in
[`src/lithos_ssl/data/splits.py`](src/lithos_ssl/data/splits.py) for the full story of
how this was figured out.

1. Install the Kaggle CLI (already in `requirements.txt`) and set up your API token:
   Kaggle account -> Settings -> "Create New Token" -> put the credentials at
   `~/.kaggle/kaggle.json` as `{"username": "...", "key": "..."}` (`chmod 600` it).
   Full instructions: https://github.com/Kaggle/kaggle-api#api-credentials
2. Download and unzip -- **this is large, about 18.6 GB**, so make sure you have disk
   space and a bit of time:
   ```bash
   kaggle datasets download -d paolaruizpuentes/lithos-dataset -p data/raw/lithos --unzip
   ```
3. Build the prepared index and sanity-check it, **before** launching any real
   training run:
   ```bash
   python scripts/prepare_data.py --root data/raw/lithos
   ```
   This reads all ~105,802 per-patch label files once (a progress bar is shown; it'll
   take a few minutes), writes `data/raw/lithos/patch_index.csv`, then reports the
   class distribution and the train/val/test split sizes. If anything about the
   download doesn't match what's documented above (a rare future dataset-version
   change, for instance), the error messages point at the exact constants to fix in
   `src/lithos_ssl/data/splits.py` -- everything downstream reads those same
   constants, so it's a one-place fix.

DINOv3's pretrained weights (used only for the frozen baseline) are *also*
access-gated, separately, on Hugging Face: accept the license at
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and run
`huggingface-cli login` once before `python scripts/evaluate.py --model dinov3_frozen`
will work.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .        # makes `import lithos_ssl` work from scripts/, tests/, anywhere
pytest -q               # ~28 tests, synthetic data only, no network -- should take <30s
```

This exact dependency set (PyTorch 2.14, PyTorch Lightning 2.6.5, Lightly 1.5.26,
Hydra 1.3.6) is verified working on Apple Silicon (MPS) as of writing.

## Running an experiment

```bash
python scripts/train.py model=simclr          # fastest, good first smoke run
python scripts/train.py model=dino
python scripts/train.py model=dinov2
python scripts/train.py model=msn
```

Every piece of that command is overridable -- see
[`configs/train.yaml`](configs/train.yaml) for the full list, and
[`configs/model/<name>.yaml`](configs/model) for method-specific hyperparameters
(every YAML field maps 1:1 to that method's constructor argument, so adding a new
tunable knob is a one-line change in both places). Common overrides:

```bash
python scripts/train.py model=dinov2 trainer.max_epochs=50 data.batch_size=32
python scripts/train.py model=msn data.root=/somewhere/else
```

During training, an `OnlineKNNEvalCallback`
([`src/lithos_ssl/eval/callbacks.py`](src/lithos_ssl/eval/callbacks.py)) periodically
freezes the current backbone and measures kNN accuracy on a held-out labelled split --
this is what decides which checkpoint is "best" (`checkpoints/<method>/best.ckpt`),
instead of just picking whichever epoch had the lowest pretext loss (a lower
DINO/MSN/SimCLR loss doesn't reliably mean better downstream features). The final
epoch's checkpoint is always saved too (`checkpoints/<method>/last.ckpt`), regardless
of its kNN score, specifically so it's available for further fine-tuning later even if
it wasn't the single best epoch for this particular evaluation.

Watch training live with TensorBoard:

```bash
tensorboard --logdir results/logs
```

## Evaluating and comparing methods

```bash
python scripts/evaluate.py --model simclr --checkpoint checkpoints/simclr/best.ckpt
python scripts/evaluate.py --model dinov2 --checkpoint checkpoints/dinov2/best.ckpt
python scripts/evaluate.py --model dinov3_frozen     # no checkpoint -- frozen baseline
python scripts/aggregate_results.py                  # rebuilds results/RESULTS.md
```

Each `evaluate.py` call reports two standard SSL evaluation numbers on the held-out
test split (see [`src/lithos_ssl/eval/knn.py`](src/lithos_ssl/eval/knn.py) and
[`linear_probe.py`](src/lithos_ssl/eval/linear_probe.py) for what they mean and why
both are useful) and writes one JSON record to `results/runs/`.
`aggregate_results.py` then rebuilds `results/RESULTS.md` from every record --
regenerating the whole table from immutable per-run files, rather than editing the
markdown directly, means re-running evaluation can never corrupt or duplicate a row.

## Train/val/test splitting

Patches are split by **thin section**, not randomly by patch -- see the detailed
rationale in [`src/lithos_ssl/data/splits.py`](src/lithos_ssl/data/splits.py). Short
version: patches from the same slide look extremely similar to each other, so a random
patch-level split would leak near-duplicates across train/val/test and inflate every
accuracy number. The section id (two directory levels above the image file) was run
against the real, full 105,802-row label index while building this project and
produces exactly 580 distinct sections with zero leakage across splits -- matching the
dataset's documented section count exactly, so this isn't a guess.

## Fine-tuning on the company dataset later

Every `checkpoints/<method>/{best,last}.ckpt` is a standard PyTorch Lightning
checkpoint, loadable via `ModelClass.load_from_checkpoint(path)` (see
`scripts/evaluate.py` for exactly that). For fine-tuning, you'd typically pull the
backbone out of it (e.g. `model.teacher_backbone` for DINO/DINOv2, `model.backbone`
for MSN, `model.backbone` for SimCLR -- see each `embed()` method for which network is
the "real" representation) and attach a new task-specific head.

## Honest limitations of this round

This codebase was built and tested on a machine with no CUDA GPU and no access to the
(login-gated) DINOv3 weights. What was actually verified here:

- **The schema**: not a guess -- authenticated against the real Kaggle API, listed the
  real file tree, downloaded the real fold-index CSVs (all 105,802 real rows) and one
  real per-patch label file, and ran the actual `section_level_split`/
  `section_id_from_patch_path` logic against that real data (580 sections, zero
  leakage, matching the documented numbers exactly -- see "Getting the data" above).
- **The pipeline**: every method's training_step, checkpointing, the online-eval
  callback, linear-probe/kNN evaluation, and results aggregation run correctly end to
  end, via both the pytest suite (`tests/`, synthetic data laid out exactly like the
  real download) and a real `scripts/train.py` → `evaluate.py` → `aggregate_results.py`
  run on a synthetic dataset (`scripts/make_synthetic_dataset.py`).

What's **not** verified yet: the actual ~18.6GB of images themselves (downloading those
requires your Kaggle login, not something this session did), and therefore any real
accuracy numbers -- the synthetic run's numbers are meaningless (random noise, a couple
of epochs). The real comparison needs the real images and real training time, on real
GPU hardware, which is the natural next step from here.
