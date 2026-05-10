# Prompt ablation report — `outputs/v1/checkpoint-final`

- Run timestamp (UTC): `20260505-175401`
- Total ablations: 6

| rank | name | flags | val_acc | wall (s) |
|---:|:---|:---|---:|---:|
| 1 | drop_metadata | `--no-metadata` | 0.6145 | 543.5 |
| 2 | baseline_trained | `` | 0.6040 | 547.5 |
| 3 | minimal_no_lec_no_meta | `--no-lecture --no-metadata` | 0.5840 | 518.4 |
| 4 | drop_lecture | `--no-lecture` | 0.5706 | 523.6 |
| 5 | answer_is | `--prompt-style answer_is` | 0.5286 | 561.8 |
| 6 | answer_is_minimal | `--no-lecture --no-metadata --prompt-style answer_is` | 0.5105 | 542.4 |