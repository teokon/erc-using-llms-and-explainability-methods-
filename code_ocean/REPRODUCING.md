# Reproducing the results (Code Ocean capsule)

This capsule reproduces the paper's **explainability figures and tables** from the saved
attributions — deterministically, CPU-only, in a few minutes. Training the models and running the
explainers over the full corpus (Optimus Prime alone is ~80 s/instance) is **not** done here; those
were run once and only their artifacts are replayed. The full pipeline (training + explaining) is
documented in the repository `README.md` and runs on your own GPU hardware.

## What the capsule produces (`/results`)

| Script (in `Explainability/`) | Output | Paper item |
|---|---|---|
| `paper_plots_from_scores.py` | `figures_optimus/paper_{meld,iemocap}_{cumulative,cov10_by_class,sparsity}.png` | Cumulative contribution curves + Coverage@10% (single-FT vs RoBERTa-base) |
| `explanation_figures.py` | `figures/faithfulness_{ds}_{metrics,curves}.png`, `figures/agreement_{ds}.png` | Faithfulness (note 5) + agreement (note 7) figures |
| `explanation_agreement_matrix.py` | `agreement/agreement_matrix.md` | Cross-method Spearman ρ / top-20 % Jaccard matrices (note 7) |
| `build_note7_context.py` | `note7_agreement_context/Note7_agreement_response.md` | Note-7 agreement response (context-aware model) |

`build_reviewer_response.py` (assembles the full reviewer-response `.docx`) is **not** part of the
capsule auto-run: it needs the training-result CSVs and all figures, i.e. a full `./checkpoints`.
Run it separately with `python Explainability/build_reviewer_response.py` once those are present.

## Inputs the capsule reads (`/data`)

The scripts read only saved artifacts (no models, no raw text needed):

| `/data/…` | Contents | Used by |
|---|---|---|
| `faith_final/scores_*.npz` | per-token attributions, single-FT + pretrained | `paper_plots_from_scores`, `explanation_agreement_matrix` |
| `faith_context/scores_*.npz` | per-token attributions, context-aware model | `explanation_agreement_matrix`, `build_note7_context` |
| `faithfulness/faithfulness_{ds}_summary.csv` (+ curve CSVs) | ERASER metric summaries | `explanation_figures` |
| `agreement/agreement_{ds}_*.{csv,json}` | agreement per-example + summary | `explanation_figures`, `build_note7_context` |
| `figures/*.png` | pre-rendered metric/agreement PNGs | `build_reviewer_response` (optional) |

## How to run

Locally (outside Code Ocean), from the repository root:

```bash
pip install -r code_ocean/environment/requirements.txt
ERC_CHECKPOINTS=./checkpoints ERC_RESULTS=./results bash code_ocean/run
```

> **Code Ocean note:** the environment image is built *before* `/code` is mounted, so the capsule's
> `environment/postInstall` cannot `pip install -r /code/...`. Install the packages directly there —
> see `code_ocean/environment/postInstall` (`pip install pandas scipy matplotlib python-docx`).

In a Code Ocean capsule the entrypoint is `code_ocean/run`; it defaults to `ERC_CHECKPOINTS=/data`
and `ERC_RESULTS=/results`. To assemble the `/data` folder from a full `checkpoints/` directory, use:

```bash
bash code_ocean/stage_data.sh ./checkpoints   # copies the needed subset into ./capsule_data
```

then attach `capsule_data/` as the capsule's `/data`.

## Notes

- Paths are resolved by `erc_paths.py` from the `ERC_CHECKPOINTS` / `ERC_RESULTS` environment
  variables, so nothing is machine-specific.
- The capsule needs no GPU and no `torch`/`transformers`; see `environment/requirements.txt`.
- Small numeric differences from the printed figures are not expected — the artifacts are fixed —
  but figure styling may vary slightly with the matplotlib version.
