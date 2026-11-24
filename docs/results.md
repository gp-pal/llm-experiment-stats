## Accuracy per arm

Truncated replies were cut off at the token limit; they are scored like any other.

| Arm | Model | Items | Correct | Accuracy | 95% CI (Wilson) | Wrong number | Bad format | Truncated |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `baseline` | openai/gpt-oss-20b | 200 | 159 | 79.5% | 73.4% to 84.5% | 40 | 1 | 1 |
| `baseline-repeat` | openai/gpt-oss-20b | 200 | 158 | 79.0% | 72.8% to 84.1% | 40 | 2 | 2 |
| `decompose` | openai/gpt-oss-20b | 200 | 114 | 57.0% | 50.1% to 63.7% | 69 | 17 | 18 |

## Planned comparisons

Difference is accuracy of B minus accuracy of A, on the items both arms answered.

| A | B | Items | Only A right | Only B right | Difference | 95% CI (bootstrap) | p (McNemar) | p (Holm) | Verdict |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `baseline` | `baseline-repeat` | 200 | 3 | 2 | -0.5 pp | -2.5 pp to +1.5 pp | 1.0000 | 1.0000 | no difference detected |
| `baseline` | `decompose` | 200 | 58 | 13 | -22.5 pp | -30.0 pp to -15.0 pp | < 0.0001 | < 0.0001 | `baseline` is better |

### Where `baseline` and `baseline-repeat` differ

Descriptive only: these slices were not planned as tests and are small.

| Slice | Items | `baseline` | `baseline-repeat` | Difference |
|---|---:|---:|---:|---:|
| thousands | 100 | 86.0% | 86.0% | +0.0 pp |
| millions | 100 | 73.0% | 72.0% | -1.0 pp |
| nominative | 100 | 77.0% | 78.0% | +1.0 pp |
| accusative | 100 | 82.0% | 80.0% | -2.0 pp |
| a group is skipped | 64 | 84.4% | 82.8% | -1.6 pp |
| no group is skipped | 136 | 77.2% | 77.2% | +0.0 pp |

### Where `baseline` and `decompose` differ

Descriptive only: these slices were not planned as tests and are small.

| Slice | Items | `baseline` | `decompose` | Difference |
|---|---:|---:|---:|---:|
| thousands | 100 | 86.0% | 54.0% | -32.0 pp |
| millions | 100 | 73.0% | 60.0% | -13.0 pp |
| nominative | 100 | 77.0% | 61.0% | -16.0 pp |
| accusative | 100 | 82.0% | 53.0% | -29.0 pp |
| a group is skipped | 64 | 84.4% | 59.4% | -25.0 pp |
| no group is skipped | 136 | 77.2% | 55.9% | -21.3 pp |

## Without truncated replies

Not a planned test: a check of how much of each difference is replies cut off at
the token limit. Items where either arm's reply was truncated are left out.

| A | B | Items | Only A right | Only B right | Difference | 95% CI (bootstrap) |
|---|---|---:|---:|---:|---:|---|
| `baseline` | `baseline-repeat` | 197 | 3 | 2 | -0.5 pp | -2.5 pp to +1.5 pp |
| `baseline` | `decompose` | 181 | 47 | 13 | -18.8 pp | -26.5 pp to -11.0 pp |
