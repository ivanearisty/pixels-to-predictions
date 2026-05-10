# Failure analysis -- v1 on val (`v1-val.npz`)

- Source logits: `results/logits/v1-val.npz`
- Source labels: `data/val.csv`
- Overall accuracy: **640/1048 = 0.6107**
- Slice families analyzed: 6

## Biggest opportunities (top 5 high-volume, low-accuracy slices)

Slices restricted to those with n >= 30; ranked by accuracy ascending then size descending.

| rank | slice_type | slice | n | correct | accuracy |
|---:|:---|:---|---:|---:|---:|
| 1 | num_choices | 5 | 44 | 9 | 0.2045 |
| 2 | topic | chemistry | 91 | 42 | 0.4615 |
| 3 | image_archetype | other | 364 | 169 | 0.4643 |
| 4 | grade | grade8 | 231 | 125 | 0.5411 |
| 5 | topic | earth-science | 81 | 44 | 0.5432 |

## by num_choices

| slice | n | correct | accuracy |
|:---|---:|---:|---:|
| 5 | 44 | 9 | 0.2045 |
| 3 | 508 | 290 | 0.5709 |
| 4 | 252 | 171 | 0.6786 |
| 2 | 244 | 170 | 0.6967 |

## by subject

| slice | n | correct | accuracy |
|:---|---:|---:|---:|
| language science | 28 | 10 | 0.3571 |
| natural science | 777 | 456 | 0.5869 |
| social science | 243 | 174 | 0.7160 |

## by topic

| slice | n | correct | accuracy |
|:---|---:|---:|---:|
| writing-strategies | 21 | 6 | 0.2857 |
| literacy-in-science | 5 | 2 | 0.4000 |
| chemistry | 91 | 42 | 0.4615 |
| earth-science | 81 | 44 | 0.5432 |
| physics | 213 | 118 | 0.5540 |
| geography | 129 | 72 | 0.5581 |
| reading-comprehension | 7 | 4 | 0.5714 |
| biology | 273 | 173 | 0.6337 |
| world-history | 11 | 7 | 0.6364 |
| science-and-engineering-practices | 114 | 77 | 0.6754 |
| us-history | 38 | 33 | 0.8684 |
| economics | 64 | 61 | 0.9531 |
| civics | 1 | 1 | 1.0000 |

## by grade

| slice | n | correct | accuracy |
|:---|---:|---:|---:|
| grade12 | 8 | 0 | 0.0000 |
| grade8 | 231 | 125 | 0.5411 |
| grade7 | 182 | 109 | 0.5989 |
| grade3 | 106 | 65 | 0.6132 |
| grade2 | 36 | 23 | 0.6389 |
| grade4 | 172 | 110 | 0.6395 |
| grade6 | 192 | 123 | 0.6406 |
| grade5 | 119 | 83 | 0.6975 |
| grade1 | 1 | 1 | 1.0000 |
| grade10 | 1 | 1 | 1.0000 |

## by hint richness

| slice | n | correct | accuracy |
|:---|---:|---:|---:|
| no_rich_hint | 434 | 265 | 0.6106 |
| rich_hint | 614 | 375 | 0.6107 |

## by image archetype

| slice | n | correct | accuracy |
|:---|---:|---:|---:|
| other | 364 | 169 | 0.4643 |
| photo_natural | 349 | 222 | 0.6361 |
| diagram_small_square | 3 | 2 | 0.6667 |
| wide_banner | 195 | 137 | 0.7026 |
| diagram_fixed_square | 137 | 110 | 0.8029 |
