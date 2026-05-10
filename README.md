# Pixels to Predictions

SmolVLM-500M-Instruct fine-tune for the NYU DL Spring 2026 "Pixels to Predictions" Kaggle final.
Multimodal multiple-choice science QA (image + question + choices → answer index).

> **Note on publication.** We realized late that this repository was never pushed to GitHub during the active competition window — development happened locally and we did not configure a remote until after the submission deadline. The code in this single-commit snapshot is the state that produced the submitted predictions; only the publication timeline is out of order. Apologies for the inconsistency between the report's reproducibility statement and the actual push date.

## Competition constraints

| | |
| --- | --- |
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` only |
| External data | None — competition data only |
| Trainable parameter cap | **5 M** (LoRA / adapters / classifier head included) |
| Eval environment | Offline, Kaggle / Colab Free tier |
| Deadline | 14 days from kickoff |
| Submission | `submission.csv` with `id,answer` (0-indexed) |

## Primary compute target

Advantech AGX **Thor** (aarch64, Jetson Thor Blackwell GPU, 128 GB unified memory) on the
`agxthor` Tailscale host. Two secondary compute paths:

- **kepler** (RTX 2000 Ada 16 GB, x86_64) — local dev + sanity checks
- **Kaggle / Colab** — final offline-eval environment, short runs only

Every trainable-parameter count is enforced by `pixels_to_predictions.budget` so we can't
accidentally blow the 5 M cap regardless of where we train.

## Project layout

```
pixels-to-predictions/
├── data/                          # competition data (gitignored, extracted from zip)
│   ├── train.csv  val.csv  test.csv  sample_submission.csv
│   └── images/{train,val,test}/*.png
├── outputs/                       # HF Trainer checkpoints (gitignored)
│   └── {run_name}/checkpoint-N/
├── results/
│   ├── submissions/               # Kaggle submission CSVs
│   ├── ablations/                 # one JSON per experiment
│   └── figures/                   # publication plots
├── logs/                          # *.log + *.pid files (per-run)
├── planning/journal.md            # experiment diary (git-tracked)
├── references/                    # competition PDF + starter notebooks (gitignored)
├── configs/                       # YAML search-space + training-preset configs
├── docker/                        # Thor-specific Dockerfile
├── scripts/                       # one-off entry points
├── stubs/                         # type stubs for libs without py.typed
└── src/pixels_to_predictions/
    ├── config.py                  # TrainingConfig, LoRAConfig, GenerationConfig
    ├── data.py                    # CSV + image loading, prompt formatting, MCQ choice letters
    ├── model.py                   # SmolVLM loader + LoRA wrapper
    ├── budget.py                  # 5M-trainable-param invariant, param audit
    ├── train.py                   # SFTTrainer setup + main() entry
    ├── predict.py                 # test → submission.csv
    ├── evaluate.py                # validation accuracy + per-subject breakdown
    ├── seed.py                    # deterministic run helpers
    ├── report.py                  # post-run metric aggregation
    └── search/                    # overnight optimizer (Karpathy-autoresearch style)
        ├── __main__.py            # CLI: `python -m pixels_to_predictions.search run ...`
        ├── experiment.py          # Experiment dataclass + JSONL trial records
        ├── space.py               # SearchSpace (grid / random sampling)
        ├── scheduler.py           # RandomScheduler, GridScheduler, ASHAScheduler
        ├── runner.py              # subprocess orchestration per trial
        ├── trials.py              # trial registry read/write
        └── summary.py             # end-of-run markdown + plots
```

## Getting started

Prereqs: `uv`, Python 3.11, CUDA driver, ~5 GB free disk.

```bash
# 1. Create venv + install dev tooling
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,search]"

# 2. Install the right GPU stack for the machine you're on
# Thor (aarch64 + CUDA 13):
uv pip install -e ".[gpu-thor]" --index-strategy unsafe-best-match
# x86_64 + CUDA 12 (Colab / Kaggle / kepler):
uv pip install -e ".[gpu-x86]"

# 3. Extract competition data into data/
python scripts/setup_data.py --zip ~/WorkDir/pixels-to-predictions.zip

# 4. Sanity check: load one sample end-to-end through the model
python scripts/sanity_check.py

# 5. Train a baseline (single config)
python -m pixels_to_predictions.train --run-name baseline

# 6. Generate a submission
python -m pixels_to_predictions.predict \
    --checkpoint outputs/baseline/checkpoint-final \
    --out results/submissions/baseline.csv

# 7. Overnight search across LoRA-r / LR / epochs
bash scripts/overnight.sh   # 8-hour budget, writes to logs/search/<timestamp>/
```

## Overnight optimizer

The `search/` subpackage is an unattended hyperparameter optimizer inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). Given a search space
and a wall-clock budget, it:

1. Samples configs from the search space (random / grid / ASHA).
2. Spawns one training subprocess per trial, captured to its own log file.
3. Writes one JSONL line per trial to `logs/search/<run>/trials.jsonl`.
4. At the end (or on SIGINT), emits a markdown report with a ranked table,
   best-trial config, and per-hyperparameter sensitivity plots.

```bash
python -m pixels_to_predictions.search run \
    --space configs/search/overnight_v1.py \
    --budget 8h \
    --max-parallel 1 \
    --out logs/search/overnight-$(date +%Y%m%d)
```

The final report lands at `logs/search/<run>/report.md` with a symlinked `best/`
checkpoint pointing at the winning trial's outputs.

## Quality gates

Before any commit:
```bash
ruff check .
ruff format --check .
mypy src/pixels_to_predictions
```

The CI intent is "green ruff + green mypy", not a test suite — competitions are
validated by leaderboard, not pytest.

## References

- [Competition page](https://www.kaggle.com/competitions/pixels-to-predictions)
- `references/` (gitignored) holds the PDF spec + starter notebook; ask if you don't have them.
- `planning/journal.md` — running experiment diary, decisions, and results.
