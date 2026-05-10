# CLAUDE.md — project-specific context for Claude Code

## Project

NYU DL Spring 2026 **Pixels to Predictions** Kaggle final. Fine-tune
`HuggingFaceTB/SmolVLM-500M-Instruct` on multimodal MCQ (image + question + choices →
0-indexed answer). Primary compute: **Advantech AGX Thor** (Jetson Thor Blackwell,
aarch64, 128 GB unified memory) at Tailscale host `agxthor`.

## Hard constraints (read every time)

- **Only** `HuggingFaceTB/SmolVLM-500M-Instruct` from HF Hub. No other pretrained models.
- **Only** the provided competition CSVs + images. No external data, no scraped corpora.
- **<= 5 M trainable parameters total.** LoRA weights + head + any adapter. Enforced by
  `pixels_to_predictions.budget.assert_under_budget()` at model init — do NOT remove
  or weaken that check.
- Evaluation environment is **offline**, Kaggle/Colab free tier.
- Submission schema: `submission.csv` with columns `id,answer` where `answer` is the
  0-indexed choice integer.

## Style

- Follow the svg-gen conventions: strict mypy, ruff ALL with ML carveouts, dataclass
  configs, JSON-per-experiment instead of wandb. See sibling project
  `~/WorkDir/ivan-training/svg-gen/` for reference.
- `uv` for all package management; never `pip install` directly.
- Don't add new top-level directories without a reason; the layout in README.md is
  intentional.
- Keep `data/`, `outputs/`, `results/submissions/*.csv`, `logs/*.log` out of git. Commit
  training code, stubs, configs, journal, and the scaffold.

## Workflow

1. Before any change, check `planning/journal.md` for current experimental state.
2. When training, always go through `pixels_to_predictions.train.main()` — do not write
   ad-hoc scripts that bypass `budget.assert_under_budget()`.
3. After each experiment worth remembering, append a dated entry to
   `planning/journal.md` with config, result, and what you'll try next.

## Useful commands

```bash
# type-check + lint
mypy src/pixels_to_predictions && ruff check .

# one-sample smoke test
python scripts/sanity_check.py

# overnight sweep
bash scripts/overnight.sh

# dry-run a trial config without training
python -m pixels_to_predictions.train --run-name dryrun --max-steps 2
```

## When you're unsure

Ask before:
- modifying `budget.py` (changes the parameter invariant)
- adding new dependencies to `pyproject.toml`
- submitting to Kaggle (we have limited daily submissions)
