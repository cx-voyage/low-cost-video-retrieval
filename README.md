# Low-Cost Video Retrieval

**Zero training, zero annotation, CPU only.** A text query goes in, ranked video clips come out.

This repository is the code and measurement data behind the report *"How Cheap Can
Video Retrieval Be? A Zero-Shot, CPU-Only Cost--Accuracy Study"*. It exists to answer one
question with numbers: if you have no GPU, no training budget and no annotated data, how
much retrieval quality is left, and where does the remaining cost actually go?

Everything is built from a **frozen** OpenCLIP ViT-B/32 encoder, **one frame per second**,
and **exhaustive cosine search**. Nothing is trained or fine-tuned.

**Report (PDF):** [`report/main.pdf`](report/main.pdf) · **Every number:**
[`results/results.csv`](results/results.csv) · **Audit:**
[`code/audit_numbers.py`](code/audit_numbers.py)

---

## Results

MSR-VTT test 1k, 1 fps, max aggregation over frames ($n = 1{,}000$ queries):

| Metric | This system | Random baseline |
|---|---|---|
| Recall@1 | **32.2 %** | 0.1 % |
| Recall@5 | 55.6 % | 0.5 % |
| Recall@10 | 66.5 % | 1.0 % |

Indexing cost on a 4-core laptop CPU (Intel i5-1155G7, no discrete GPU):

| Quantity | Value |
|---|---|
| Frames indexed | 16,036 (16.0 per clip) |
| Build time | 1,035.8 s (65 ms per frame) |
| Index size | 32.84 MB (16,036 × 512 × 4 B) |
| Query cost | 60–100 ms text encoding + 1–6 ms search |

Two facts worth knowing when reading these numbers:

* 32.2 % Recall@1 is within 1.5 pp of the 30.7 % that a frozen zero-shot CLIP ViT-B/32 baseline
  reports on the same 1k split (Ventura, Schmid & Varol, *Learning text-to-video retrieval from
  image captioning*, arXiv:2404.17498, Table 5), and four independent checks (frame count,
  implied hit count, index size, random baseline) agree with the raw predictions. Supervised
  systems are far above this; the trade is deliberate.
* Changing the sampling rate from 0.25 to 3.0 fps — the ceiling, because the stored clips are
  themselves 3 fps — moves Recall@1 by 1.7 pp (paired $p = 0.12$) while changing build cost by
  14×. Recall@10 does fall by 2.4 pp at 0.25 fps (uncorrected $p = 0.03$). **Frame rate is not the
  bottleneck; the encoder is.**

> `results/results.csv` is the single source of truth for every number above. If anything
> in this README, the figure or the paper disagrees with it, the CSV is correct.

---

## How it works

```
offline, once per clip          online, once per query

Video clip → Sample 1 fps → ViT-B/32 (frozen) → N×512 matrix
                                                      ↓
Text query → Text tower ──────→ cosine, max over frames → top-K clips
```

1. **Decode** every clip at 1 fps (one frame out of thirty).
2. **Encode** each frame into a 512-d L2-normalised vector with the frozen encoder.
3. **Search** by one matrix–vector product, reduce frame scores to clip scores by taking
   the maximum, and rank.

No shot detection, no keyframe extraction, no index structure, no temporal model. The
index is a single float32 matrix; a query is one dot product.

---

## Repository layout

```
code/        Python scripts (download, index, search, evaluate, benchmark, plot)
results/     results.csv, per-run metrics (JSON), timing CSVs, the cost–accuracy figure
report/      LaTeX source, bibliography and compiled PDF of the paper
log.md       raw experiment log, append-only (every run, including failed ones)
env.ps1      activates the project virtual environment
```

---

## Requirements

| Item | Value |
|---|---|
| Python | 3.11 |
| Hardware | any x86-64 CPU, 4 threads used; **no GPU required** |
| RAM | 16 GB is comfortable; 8 GB works with `--max-clips` |
| Disk | ~3.7 GB for the datasets (2.7 GB) and the encoder weights (577 MB) |

Datasets and model weights are **not** committed to this repository — they are downloaded
by the scripts below.

---

## Quick start

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 2. data (UCF-101 subset, ~171 MB, one-time)
python code/prepare_data.py

# 3. index (first run also downloads the CLIP weights, ~350 MB)
python code/build_index.py --fps 1 --out-dir data\index

# 4. query
python code/search.py "a person playing basketball"

# 5. evaluate Recall@K against the random baseline
python code/evaluate.py --topk 10
```

### Adding MSR-VTT (the dataset used in the paper)

```powershell
python code/download_msrvtt.py     # 2.19 GB, resumable; re-run after an interruption
python code/prepare_msrvtt.py      # unpack videos + captions
python code/build_index.py --fps 1 --out-dir data\index_msrvtt
python code/evaluate.py --protocol msrvtt --index-dir data\index_msrvtt --topk 10 `
    --save results/msrvtt_vitb32_1fps.json
```

The dataset is inferred from the output directory name: any `--out-dir` containing
`msrvtt` reads videos from `data\msrvtt\video`, otherwise from `data\ucf101`.

### Reproducing the ablations

```powershell
# Sampling rate. The stored MSR-VTT clips are themselves 3 fps, so the achievable
# ladder is 3, 1.5, 1, 0.5, 0.25 fps (step = round(3 / fps)); asking for 4 fps
# saturates at 3 fps and build_index.py now warns about it.
python code/build_index.py --fps 3    --max-frames 90 --out-dir data\index_msrvtt_fps3
python code/build_index.py --fps 1.5  --out-dir data\index_msrvtt_fps1.5
python code/build_index.py --fps 0.5  --out-dir data\index_msrvtt_fps0.5
python code/build_index.py --fps 0.25 --out-dir data\index_msrvtt_fps0.25

# temporal aggregation: replace max with mean
python code/evaluate.py --protocol msrvtt --index-dir data\index_msrvtt --agg mean `
    --save results/msrvtt_vitb32_1fps_mean.json

# append a row to results.csv (the script checks fields and refuses to overwrite)
python code/record_result.py --eval results/msrvtt_vitb32_1fps.json `
    --index data\index_msrvtt --build-seconds 1035.8

# component-level timing: decoding vs encoder, measured separately
python code/bench_decompose.py --dataset msrvtt --clips 20

# regenerate the cost–accuracy figure
python code/make_figure.py

# query-type stratification (paper Table V) and budget/scale derivations
python code/analyze_query_types.py
python code/analyze_budget_scale.py

# paired McNemar tests behind every p-value in the paper -> results/significance.csv
python code/analyze_significance.py

# audit: every number in the paper's tables and derived claims vs results/*.csv
python code/audit_numbers.py
```

---

## What the paper reports, and where it comes from

| Paper element | Source |
|---|---|
| Table: sampling-rate ablation | `results/results.csv`, rows `fps 0.25/0.5/1/1.5/3` |
| Table: main results (Recall@K) | `results/msrvtt_vitb32_1fps.json` |
| Table: system configuration | `code/build_index.py` defaults, `env.ps1` |
| Decoding vs encoder split (5.6 / 40.4 ms) | `results/timing_components.csv` |
| Run-to-run spread (197.6 / 207.1 / 251.1 s) | `results/timing_repeats.csv` |
| Cost–accuracy figure | `results/fig_cost_accuracy.png` via `code/make_figure.py` |
| Table: query-type stratification | `results/query_types.csv` via `code/analyze_query_types.py` |
| Budget / scale derivations (crossover prices, 7.7e6-frame memory wall, 17.6× at P=50) | `results/derived_budget_scale.csv` via `code/analyze_budget_scale.py` |

---

## Design decisions, and what was deliberately left out

| Decision | Why | What it costs |
|---|---|---|
| Frozen encoder, no training | removes training data, GPU training and annotation entirely | ~zero-shot accuracy, below supervised systems |
| 1 frame per second | encoder calls are the dominant cost; 30 fps costs 30× more | events shorter than 1 s can be missed |
| One vector per frame (no objects) | one encoder call per frame instead of thousands | no localisation, no object-level queries |
| Exhaustive search, no ANN index | no index to build, no parameters to tune | linear scan — fine to ~10⁷ frames |
| No product quantisation | compute dominates storage at this scale | index is 4 bytes per dimension |

The first two rows are measured, not asserted: see the sampling-rate ablation and
`results/timing_components.csv`.

---

## Known limitations

* **Motion is discarded.** Frame-level embeddings cannot distinguish "a person jumps" from
  "a person stands"; a query for a dunk returns generic basketball clips, with scores
  compressed into a 0.010 band.
* **Temporal resolution is 1 s**, so no precise localisation of events.
* **Single stage.** 34.3 pp of queries place the correct clip between ranks 2 and 10, so a
  re-ranker over the top-10 could lift Recall@1 from 32.2 % to 66.5 %.
* **Scale ceiling.** Beyond roughly 10⁷ frames (3,200–4,100 hours of video), exhaustive
  search and CPU-only indexing stop being appropriate; the same pipeline should move to a
  GPU rather than be redesigned.
* **Zero-shot only.** No fine-tuning, so domain-specific corpora will underperform.

---

## Report

* `report/main.pdf` — 7 pages, IEEE Transactions style
* `report/main.tex` — LaTeX source (pipeline figure drawn with TikZ)
* `report/refs.bib` — bibliography

Build:

```powershell
cd report
pdflatex main ; bibtex main ; pdflatex main ; pdflatex main
```

---

## Troubleshooting

* **`pip install` is very slow** — keep the `-i https://mirrors.aliyun.com/pypi/simple/`
  mirror flag shown above.
* **Model download fails** — the scripts set `HF_ENDPOINT=https://hf-mirror.com`. If it
  still fails, check that no other configuration overrides that variable.
* **Some videos cannot be decoded** — OpenCV does not support every codec; the script
  prints a line for each clip it cannot read and continues with the rest.
* **Out of memory** — cap the corpus with `--max-clips 300`.

---

## Contact

Chengxuan Xie — Nanchang University — chengxuan_xie@email.ncu.edu.cn
