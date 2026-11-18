# code/

All usage instructions — install, download data, index, query, evaluate, ablations — live in
the repository root:

**[../README.md](../README.md)**

This folder contains only the scripts. Quick map:

| Script | What it does |
|---|---|
| `prepare_data.py` | download + unpack the UCF-101 subset |
| `download_msrvtt.py` | download the MSR-VTT video archive (resumable) |
| `prepare_msrvtt.py` | unpack MSR-VTT videos and captions |
| `build_index.py` | sample frames + encode with a frozen CLIP + save the vector matrix |
| `search.py` | one text query → top-k clips |
| `evaluate.py` | Recall@K with the correct random baseline |
| `record_result.py` | append a run to `results/results.csv` (never overwrites) |
| `bench_decompose.py` | time video decoding and the encoder separately |
| `make_figure.py` | regenerate the cost–accuracy figure |
| `dl.py`, `get_model.py` | download helper and model loader |

`log.md` at the repository root is the raw, append-only experiment log; `results/results.csv`
is the authoritative table of numbers quoted in the report.
