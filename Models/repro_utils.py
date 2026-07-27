#!/usr/bin/env python3
"""
Reproducibility-report helper shared by the EmoBERTa-style training files.

Emits, per run, the exact experimental details an IEEE reviewer typically asks
for but that are hard to recover after the fact:
  - train/val/test preprocessing (rows kept, label distribution per split)
  - maximum-context-length usage statistics (token-length distribution)
  - truncation frequency (examples hitting the max-length cap; targets truncated)
  - hardware (GPU / CUDA / torch / library versions)
  - runtime cost (LR search, final training, total wall-clock)
  - model / hyperparameter config

Writes OUTPUT_DIR/repro_report.json (machine-readable) and
OUTPUT_DIR/repro_report.md (a paper-ready markdown table).

Usage (at the end of a training script/notebook, once datasets + timings exist):

    import repro_utils
    repro_utils.save_repro_report(
        OUTPUT_DIR,
        config={...},
        splits={
            "train": {"dataset": train_ds_full, "df": train_df},
            "val":   {"dataset": val_ds_full,   "df": val_df},
            "test":  {"dataset": test_ds_full,   "df": test_df},
        },
        tokenizer=tok, id2label=id2label, max_len=MAX_LEN,
        speaker_col=SPEAKER_COL, text_col=TEXT_COL, label_col=LABEL_COL, labels=LABELS,
        timings={"optuna_search_sec": ..., "final_training_sec": ..., "total_sec": ...},
        # sep_pairs_per_boundary=2 for EmoBERTa's </s></s>; reserved tokens = cls + eos + 2*2 seps
    )
"""

import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def hardware_info():
    n = torch.cuda.device_count()
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version() if torch.cuda.is_available() else None),
        "gpu_count_visible": n,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(n)],
        "gpu_total_mem_gb": [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1) for i in range(n)],
    }
    try:
        import transformers, datasets  # noqa
        info["transformers_version"] = transformers.__version__
        info["datasets_version"] = datasets.__version__
    except Exception:
        pass
    return info


def length_stats(lengths, max_len):
    a = np.asarray(lengths, dtype=int)
    at_cap = int((a >= max_len).sum())
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "mean": round(float(a.mean()), 1),
        "median": int(np.median(a)),
        "p90": int(np.percentile(a, 90)),
        "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
        "max_len_cap": int(max_len),
        "n_at_cap": at_cap,
        "pct_at_cap": round(100.0 * at_cap / max(a.size, 1), 2),
    }


def target_truncation_stats(df, tokenizer, max_len, speaker_col, text_col,
                            label_col, labels, sep_tokens_reserved=4):
    """Count target utterances whose own tokens exceed the budget available to
    the target segment (max_len - CLS - EOS - the separator tokens bracketing it).
    For EmoBERTa </s></s> bracketing, the target is delimited by 4 separator
    tokens, and CLS + EOS reserve 2 more, so the budget is max_len - sep - 2."""
    budget = max_len - sep_tokens_reserved - 2
    d = df.copy()
    d[label_col] = d[label_col].astype(str).str.strip().str.lower()
    d = d[d[label_col].isin(labels)]
    n = len(d)
    n_trunc = 0
    for s, u in zip(d[speaker_col].astype(str).str.upper(), d[text_col].astype(str)):
        if len(tokenizer.encode(f"{s}: {u}", add_special_tokens=False)) > budget:
            n_trunc += 1
    return {
        "n": int(n),
        "target_token_budget": int(budget),
        "n_target_truncated": int(n_trunc),
        "pct_target_truncated": round(100.0 * n_trunc / max(n, 1), 3),
    }


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _write_markdown(path, report):
    cfg = report["config"]
    hw = report["hardware"]
    rt = report.get("runtime_sec", {})
    lines = [f"# Reproducibility report — {cfg.get('run_name', 'run')}", ""]

    # --- config ---
    lines += ["## Configuration", ""]
    lines.append(_md_table(
        ["Field", "Value"],
        [[k, v] for k, v in cfg.items()],
    ))
    lines.append("")

    # --- preprocessing + context length + truncation ---
    lines += ["## Preprocessing, context-length usage & truncation (per split)", ""]
    hdr = ["Split", "N examples", "Len min", "Len mean", "Len median", "Len p95",
           "Len max", "Cap", "% at cap", "% target truncated"]
    rows = []
    for name, e in report["splits"].items():
        cl = e["context_length"]
        tt = e.get("target_truncation", {})
        rows.append([
            name, e["n_examples"], cl["min"], cl["mean"], cl["median"], cl["p95"],
            cl["max"], cl["max_len_cap"], cl["pct_at_cap"],
            tt.get("pct_target_truncated", "n/a"),
        ])
    lines.append(_md_table(hdr, rows))
    lines.append("")

    # --- label distribution ---
    lines += ["## Label distribution (per split)", ""]
    all_labels = sorted({l for e in report["splits"].values() for l in e["label_distribution"]})
    hdr = ["Split"] + all_labels + ["Total"]
    rows = []
    for name, e in report["splits"].items():
        ld = e["label_distribution"]
        rows.append([name] + [ld.get(l, 0) for l in all_labels] + [e["n_examples"]])
    lines.append(_md_table(hdr, rows))
    lines.append("")

    # --- hardware ---
    lines += ["## Hardware & software", ""]
    lines.append(_md_table(["Field", "Value"], [[k, v] for k, v in hw.items()]))
    lines.append("")

    # --- runtime ---
    if rt:
        lines += ["## Runtime cost (wall-clock)", ""]
        rows = []
        for k, v in rt.items():
            secs = float(v)
            rows.append([k, f"{secs:.0f}", f"{secs/60:.1f}", f"{secs/3600:.2f}"])
        lines.append(_md_table(["Phase", "seconds", "minutes", "hours"], rows))
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_repro_report(out_dir, *, config, splits, tokenizer, id2label, max_len,
                      speaker_col, text_col, label_col, labels,
                      timings=None, sep_tokens_reserved=4):
    """Assemble and write repro_report.json + repro_report.md under out_dir.

    splits: dict split_name -> {"dataset": built_hf_dataset, "df": raw_dataframe_or_None}
    """
    report = {"config": dict(config), "hardware": hardware_info(), "splits": {}}
    if timings:
        report["runtime_sec"] = {k: float(v) for k, v in timings.items()}

    for name, d in splits.items():
        ds = d["dataset"]
        df = d.get("df")
        lengths = [len(x) for x in ds["input_ids"]]
        labs = Counter(int(x) for x in ds["labels"])
        entry = {
            "n_examples": len(ds),
            "context_length": length_stats(lengths, max_len),
            "label_distribution": {id2label[k]: int(v) for k, v in sorted(labs.items())},
        }
        if df is not None:
            entry["target_truncation"] = target_truncation_stats(
                df, tokenizer, max_len, speaker_col, text_col, label_col, labels,
                sep_tokens_reserved=sep_tokens_reserved,
            )
        report["splits"][name] = entry

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repro_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "repro_report.md", report)
    print("[saved]", out_dir / "repro_report.json")
    print("[saved]", out_dir / "repro_report.md")
    return report
