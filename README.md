# LoopFace: When Does Test-Time Recurrence Help a Lightweight Face Detector?

LoopFace adds one **weight-tied 3×3 block** to a RetinaFace MobileNet-0.25 face detector and applies it
**L times**. The model is trained with L drawn at random from {1, …, 6}, so at test time L can be set to
any value (1, 2, 4, 8, 16, …) with the same weights. The more steps, the more compute.

We compare five placements of the loop on WIDER FACE and analyse what the loop actually changes:
false alarms by type, recall by face size, precision–recall and cost.

> Paper submitted to ICEEICT 2027 (under review).

---

## Main findings

| Model (80 epochs) | Params (M) | GMACs | Hard AP |
|---|---|---|---|
| Detector without SSH | 0.32 | 0.69 | 64.94 ± 0.16 |
| SSH baseline | 0.43 | 0.98 | 69.46 ± 0.20 |
| Loop replacing SSH, L = 1 | 0.36 | 1.00 | 66.12 ± 0.05 |
| Loop replacing SSH, L = 4 | 0.36 | 1.93 | 67.45 ± 0.36 |

(mean ± std over seeds 0, 1, 7; full tables in the paper)

- **The loop helps only when the context module (SSH) is missing.** Without SSH, Hard AP rises by
  +1.33 ± 0.34 from L = 1 to L = 4. With SSH, or before the backbone, the curve is flat.
- **The loop filters false alarms. It does not find new faces.** From L = 1 to L = 4, false alarms at
  score 0.5 fall by 27% (three seeds) and 98% of the hits remain. Recall of 8–16 px faces at score 0.02
  does not change with L.
- **One loop step costs about as much as all of SSH** (0.31 vs 0.29 GMACs) but gives less AP.

---

## Repository layout

```
object_detection/
├── train.py               # training (LoopFaceNet, random L per batch, seeds, exact resume)
├── test_widerface.py      # inference on WIDER FACE val -> widerface_evaluate/widerface_txt/<tag>/
├── detect.py              # run the detector on single images
├── data/config.py         # all model configs (cfg_mnet, cfg_mnet_loop_nossh, ...)
├── models/loopface_net.py # LoopFaceNet: backbone + FPN + (SSH | loop) + heads
├── loopface/              # loop cells (plain, dilated, coarse-to-fine, pre-backbone plain / ConvGRU)
├── layers/                # anchors (PriorBox) and MultiBoxLoss
├── utils/                 # box utils, NMS, timers
├── widerface_evaluate/    # official WIDER FACE evaluation (Easy / Medium / Hard AP)
├── analysis/
│   ├── core/              # shared code (detection matching, IoU, false-alarm types)
│   └── custom/            # one script per analysis (see "Analysis" below)
├── scripts/               # PowerShell drivers that run everything for all models
├── notebook/              # tables and figures for the paper
├── outputs/               # analysis CSVs and figures
├── runs/                  # one folder per training run (run.json, train_log.csv, final.pth)
├── weights/               # ImageNet-pretrained MobileNet-0.25 and released RetinaFace weights
├── dataset/               # WIDER FACE images and labels (not in git)
├── results.csv            # AP of every evaluated model (80 epochs)
└── results_250.csv        # AP of the 250-epoch runs
```

---

## Setup

The project uses [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`, `.python-version`).

```bash
uv sync                          # creates .venv with the pinned packages
# Windows:  .venv\Scripts\activate      Linux:  source .venv/bin/activate
```

Training needs a CUDA GPU (`MultiBoxLoss` calls `.cuda()`). Evaluation, analysis and the CPU benchmark
run without one.

Build the evaluation extension once:

```bash
cd widerface_evaluate
python setup.py build_ext --inplace
cd ..
```

### Data and weights

1. Download WIDER FACE (train + val) and place the training images and `label.txt` (RetinaFace format)
   under `dataset/train/`, i.e. `dataset/train/label.txt` next to `dataset/train/images/`.
   Place the val images where `test_widerface.py` expects them.
2. Put the ImageNet-pretrained MobileNet-0.25 backbone (`mobilenetV1X0.25_pretrain.tar` from
   [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)) in `weights/`.
3. Optional: the released RetinaFace MobileNet-0.25 weights in `weights/`, used as a reference
   (`released_mnet_L0`).

We removed 31 zero-size training boxes that cause an infinite localisation loss, leaving 12,876 images.

---

## Configs

All configs live in `data/config.py`. The loop is trained with `loop_train_L = (1, 6)`.

| Config | Placement | Paper name |
|---|---|---|
| `cfg_mnet` | none | SSH baseline |
| `cfg_mnet_nossh` | none | detector without SSH |
| `cfg_mnet_loop` | (a) after SSH | loop after SSH |
| `cfg_mnet_loop_nossh` | (b) replacing SSH | plain loop (main model) |
| `cfg_mnet_dloop_nossh` | (b) replacing SSH | dilated loop (dilation 1, 2, 4) |
| `cfg_mnet_loopcnn_nossh` / `_ssh` | (b) replacing SSH | coarse-to-fine loop (3 fixed steps) |
| `cfg_mnet_preloop` | (c) before the backbone | pre-backbone plain cell |
| `cfg_mnet_preloop_gru` | (c) before the backbone | pre-backbone ConvGRU |

`cfg_mnet_haar_loop` is an exploratory variant and is not part of the paper.

---

## Training

Each run gets its own folder `runs/<config>_s<seed>/` with `run.json` (args, config, GPU),
`train_log.csv`, `last.pth` (resumable, saved every epoch) and `final.pth`.
Re-running the same command resumes automatically from `last.pth`.

```bash
# paper schedule: 80 epochs, LR 1e-3 divided by 10 after epochs 61 and 70
python train.py --config cfg_mnet_loop_nossh --epochs 80 --seed 0
python train.py --config cfg_mnet_loop_nossh --epochs 80 --seed 1
python train.py --config cfg_mnet_loop_nossh --epochs 80 --seed 7

# full schedule: 250 epochs (decay after 190 and 220, as in the released RetinaFace)
python train.py --config cfg_mnet_loop_nossh --epochs 250 --seed 0
```

Defaults: SGD, momentum 0.9, weight decay 5e-4, LR 1e-3, batch 32 (from the config), 640×640 crops.
`--epochs N` sets the decays to 76% and 88% of N unless `--decay1/--decay2` are given.

Sanity checks:

```bash
# overfit one batch: the loss should go close to 0
python train.py --config cfg_mnet_loop_nossh --single_batch --batch_size 8 --sb_iters 500 --print_every 10
# smoke test: 100 iterations, then exit
python train.py --config cfg_mnet_loop_nossh --max_iters 100 --print_every 10
```

---

## Evaluation and analysis (all models)

The scripts in `scripts/` (PowerShell) find every run in `runs/`, evaluate each loop model at
L = 1, 2, 4, 6, 8, 12, 16, and **skip everything that is already in the CSVs**. After an interruption
(power cut etc.) just run them again. Add `-List` to only see what is left.

Run them in this order:

```powershell
.\scripts\run_all.ps1             # 1. WIDER FACE val AP           -> results.csv
.\scripts\precision_recall.ps1    # 2. precision by score, recall by size
.\scripts\false_alarm_score.ps1   # 3. false-alarm types, score separation + histogram
.\scripts\cost_pr.ps1             # 4. CPU benchmark, precision-recall curves
.\scripts\list_pending.ps1        # what is still missing, one row per run
```

Run step 4 with nothing else running on the machine, because the benchmark measures CPU time.

Evaluation tags have the form `<config>_L<L>_seed<seed>` (e.g. `cfg_mnet_loop_nossh_L4_seed0`).
Predictions go to `widerface_evaluate/widerface_txt/<tag>/`.

To evaluate a single model, see `scripts/run_all.ps1` for the exact `test_widerface.py` arguments, e.g.

```bash
python test_widerface.py -m runs/cfg_mnet_preloop_s0/final.pth --config cfg_mnet_preloop
```

### Analysis scripts

| Module (`python -m ...`) | Output | What it measures |
|---|---|---|
| `analysis.custom.precision_by_score` | `outputs/precision_by_score.csv` | hits, false alarms and precision at score 0.02 / 0.3 / 0.5 |
| `analysis.custom.recall_by_size` | `outputs/recall_by_size.csv` | recall per face-size bin (8–16 px, 16–32 px, …) |
| `analysis.custom.false_alarm_types` | `outputs/false_alarm_types.csv` | each detection classed as hit / duplicate / localisation (IoU 0.1–0.5) / background (IoU < 0.1), per 100 images |
| `analysis.custom.score_separation` | `outputs/score_separation.csv`, `outputs/score_hist.csv` | score distribution of hits vs false alarms |
| `analysis.custom.pr_curve` | `outputs/pr_curve.csv` | interpolated precision at recall 0.00, 0.01, … |
| `analysis.custom.benchmark` | `outputs/benchmark.csv` | params, GMACs (`FlopCounterMode`), CPU latency (1 thread, median of 50 runs) |

Examples:

```bash
python -m analysis.custom.false_alarm_types --score_thrs 0.3 0.5
python -m analysis.custom.recall_by_size --score_thr 0.02
python -m analysis.custom.pr_curve --tags cfg_mnet_loop_nossh_L1_seed0 cfg_mnet_loop_nossh_L4_seed0
python -m analysis.custom.benchmark --configs cfg_mnet cfg_mnet_loop_nossh --loops 1 2 4 8 16 --device cpu --threads 1
```

Notes:

- `ms_per_img` in `results.csv` includes file I/O. Use `outputs/benchmark.csv` for timing.
- AUROC at score 0.02 is dominated by low-score junk boxes; the paper reports AP and PR curves instead.

---

## Credits and license

Built on [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) (MIT License,
see `LICENSE.MIT`). The modified files say so in their headers.

## Authors

Yasin Arafat, Md. Mahibullah Howlader, Md. Samsuzzaman, Golam Md. Muradul Bashir —
Patuakhali Science and Technology University (PSTU), Bangladesh.

## Citation
Under review at ICEEICT 2027. A BibTeX entry will be added after the decision.