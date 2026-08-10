#!/usr/bin/env python3
"""Compile TextVQA full-train/full-eval benchmark summaries into one table."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "textvqa_full_results"
RAW = OUT_DIR / "raw_remote"
LOCAL_QWEN = (
    ROOT
    / "invalid_archive"
    / "20260611_old_result_artifacts_cleanup"
    / "root_dirs"
    / "qwen_results_for_codex"
    / "paper_benchmark_20260610_100729_summary.csv"
)

REMOTE_SOURCES = [
    (
        "mechanism_sweep_20260701_234542",
        RAW / "paper_benchmark_textvqa_mechanism_sweep_20260701_234542_summary.csv",
    ),
    (
        "lora_target_sweep_20260630_015549",
        RAW / "paper_benchmark_textvqa_lora_target_sweep_20260630_015549_summary.csv",
    ),
    (
        "mores_full16_20260616_233610",
        RAW / "paper_benchmark_textvqa_mores_full16_20260616_233610_summary.csv",
    ),
]

OUT_COLUMNS = [
    "source_run",
    "model",
    "dataset",
    "method",
    "method_family",
    "finetuned_scope",
    "target_modules",
    "mechanism_params",
    "train_samples",
    "eval_samples",
    "max_length",
    "metric",
    "base_score",
    "tuned_score",
    "delta",
    "avg_loss",
    "train_steps",
    "train_time_s",
    "trainable_params",
    "trainable_pct",
    "avg_prediction_words",
    "long_prediction_count",
    "top_prediction",
    "top_prediction_ratio",
    "base_similarity",
    "experiment",
    "notes",
]


TARGET_LABELS = {
    "q_proj": "attention query projection",
    "k_proj": "attention key projection",
    "v_proj": "attention value projection",
    "o_proj": "attention output projection",
    "gate_proj": "MLP gate projection",
    "up_proj": "MLP up projection",
    "down_proj": "MLP down projection",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_num(value: object) -> object:
    parsed = num(value)
    if parsed is None:
        return ""
    if parsed.is_integer():
        return int(parsed)
    return parsed


def compact_json(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def list_from_jsonish(value: str) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return []


def family_from_method(method: str, existing: str = "") -> str:
    for prefix, family in [
        ("lora", "LoRA"),
        ("qlora", "QLoRA"),
        ("dora", "DoRA"),
        ("l2t", "L2T"),
        ("mole", "MoLE"),
        ("mores", "MoReS"),
        ("reft", "ReFT"),
        ("freeze", "Freeze"),
        ("base", "Base"),
    ]:
        if method.startswith(prefix) or method == prefix:
            return family
    if existing and existing != "frozen pretrained baseline":
        return existing
    return existing or ""


def targets_to_scope(family: str, targets: list[str]) -> str:
    if not targets:
        return ""
    readable = [TARGET_LABELS.get(target, target) for target in targets]
    return f"{family} trainable adapter weights on " + "; ".join(readable)


def scope_from_params(method: str, family: str, params_text: str) -> tuple[str, str]:
    params = {}
    if params_text:
        try:
            params = json.loads(params_text)
        except json.JSONDecodeError:
            params = {}

    if family == "Freeze":
        layers = params.get("layers", "")
        modules = params.get("modules", "")
        target = str(modules).replace(",", ", ")
        return (
            f"Unfreezes original model weights in language-model layers {layers}: {target}",
            target,
        )

    if family == "ReFT":
        layers = params.get("layers", "")
        positions = params.get("positions", "")
        rank = params.get("rank", "")
        return (
            f"Trainable ReFT intervention vectors at layers {layers}, positions {positions}, rank {rank}",
            "hidden-state interventions",
        )

    if family == "MoReS":
        rank = params.get("rank", "")
        tokens = params.get("tokens", "")
        layer_range = params.get("layer_range", "")
        return (
            f"Trainable MoReS visual-token residual/router parameters, rank {rank}, tokens={tokens}, layer_range={layer_range}",
            "visual token residual routing",
        )

    targets = params.get("targets")
    if isinstance(targets, list):
        target_modules = ",".join(str(item) for item in targets)
        scope = targets_to_scope(family, [str(item) for item in targets])
        extra = []
        for key in ["experts", "rank", "alpha", "dropout"]:
            if key in params:
                extra.append(f"{key}={params[key]}")
        if extra:
            scope = f"{scope} ({', '.join(extra)})"
        return scope, target_modules

    return "", ""


def scope_from_method_name(method: str, family: str) -> tuple[str, str, str]:
    notes = ""
    if method == "base":
        return "No finetuning; pretrained baseline eval only", "", ""
    if method in {"lora", "dora", "qlora"}:
        notes = "target_modules inferred from trainable params/default config; not explicitly recorded in old summary"
        return targets_to_scope(family, ["q_proj", "v_proj"]), "q_proj,v_proj", notes
    if method.startswith("mores"):
        if "late24" in method:
            return "MoReS residual parameters on late layer 24 token positions", "visual token residual routing", ""
        if "late16" in method:
            return "MoReS residual parameters on late layer 16 token positions", "visual token residual routing", ""
        if "edges" in method:
            return "MoReS residual parameters on visual edge tokens across language layers", "visual edge token residual routing", ""
        if "wide" in method:
            return "MoReS residual parameters on wide visual-token set across language layers", "wide visual token residual routing", ""
    return "", "", ""


def full_train_eval(row: dict[str, str]) -> bool:
    exp = row.get("experiment", "")
    if row.get("method") == "base":
        return "evalfull" in exp
    return "trainfull_evalfull" in exp and row.get("eval_predictions", "") in {"5000", "5000.0", 5000}


def normalize_remote(source_run: str, row: dict[str, str]) -> dict[str, object] | None:
    if not full_train_eval(row):
        return None

    method = row.get("method", "")
    family = family_from_method(method, row.get("family", ""))
    params_text = row.get("params", "")
    scope, targets = scope_from_params(method, family, params_text)
    notes = ""
    if not scope:
        scope, targets, notes = scope_from_method_name(method, family)

    if source_run.startswith("lora_target"):
        targets_list = list_from_jsonish(row.get("targets", ""))
        family = "LoRA"
        targets = ",".join(targets_list)
        scope = targets_to_scope(family, targets_list) + " (rank=8, alpha=16, dropout=0.05)"
        params_text = compact_json({"targets": targets_list, "rank": 8, "alpha": 16, "dropout": 0.05})

    base_score = num(row.get("base_score"))
    tuned_score = num(row.get("tuned_score"))
    delta = "" if base_score is None or tuned_score is None else round(tuned_score - base_score, 4)

    return {
        "source_run": source_run,
        "model": row.get("model", "qwen25vl_3b_instruct"),
        "dataset": row.get("dataset", "textvqa"),
        "method": method,
        "method_family": family,
        "finetuned_scope": scope,
        "target_modules": targets,
        "mechanism_params": compact_json(params_text or row.get("targets", "")),
        "train_samples": "" if method == "base" else "full",
        "eval_samples": clean_num(row.get("eval_predictions")),
        "max_length": "" if method == "base" else 1536,
        "metric": row.get("metric", "vqa_accuracy"),
        "base_score": clean_num(row.get("base_score")),
        "tuned_score": clean_num(row.get("tuned_score")),
        "delta": delta,
        "avg_loss": clean_num(row.get("avg_loss")),
        "train_steps": clean_num(row.get("train_steps")),
        "train_time_s": clean_num(row.get("train_time_s")),
        "trainable_params": clean_num(row.get("trainable_params")),
        "trainable_pct": clean_num(row.get("trainable_pct")),
        "avg_prediction_words": clean_num(row.get("avg_prediction_words")),
        "long_prediction_count": clean_num(row.get("long_prediction_count")),
        "top_prediction": row.get("top_prediction", ""),
        "top_prediction_ratio": clean_num(row.get("top_prediction_ratio")),
        "base_similarity": clean_num(row.get("base_similarity")),
        "experiment": row.get("experiment", ""),
        "notes": notes,
    }


def normalize_local_qwen(row: dict[str, str]) -> dict[str, object] | None:
    method = row.get("method", "")
    if method == "freeze":
        return None
    if method != "base" and (row.get("train_samples") != "34602" or row.get("eval_samples") != "5000"):
        return None

    family = family_from_method(method, row.get("method_family", ""))
    scope, targets, notes = scope_from_method_name(method, family)
    if method == "qlora":
        notes = (notes + "; " if notes else "") + "old QLoRA full run was anomalously low"
    base_score = num(row.get("base_score"))
    tuned_score = num(row.get("tuned_score"))
    delta = "" if base_score is None or tuned_score is None else round(tuned_score - base_score, 4)

    return {
        "source_run": "qwen_full_benchmark_20260610_100729",
        "model": row.get("model", "qwen25vl_3b_instruct"),
        "dataset": row.get("dataset", "textvqa"),
        "method": method,
        "method_family": family,
        "finetuned_scope": scope,
        "target_modules": targets,
        "mechanism_params": compact_json({"targets": targets.split(",") if targets else [], "inferred": bool(targets)}),
        "train_samples": row.get("train_samples", ""),
        "eval_samples": clean_num(row.get("eval_samples")),
        "max_length": 1536 if method != "base" else "",
        "metric": row.get("metric", "vqa_accuracy"),
        "base_score": clean_num(row.get("base_score")),
        "tuned_score": clean_num(row.get("tuned_score")),
        "delta": delta,
        "avg_loss": "",
        "train_steps": "",
        "train_time_s": clean_num(row.get("train_time_s")),
        "trainable_params": clean_num(row.get("trainable_params")),
        "trainable_pct": clean_num(row.get("trainable_pct")),
        "avg_prediction_words": "",
        "long_prediction_count": "",
        "top_prediction": "",
        "top_prediction_ratio": "",
        "base_similarity": "",
        "experiment": "",
        "notes": notes,
    }


def sort_key(row: dict[str, object]) -> tuple[int, float, str]:
    score = num(row.get("tuned_score"))
    return (0 if score is not None else 1, -(score or -1.0), str(row.get("method", "")))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    display_cols = [
        "rank",
        "method",
        "method_family",
        "tuned_score",
        "trainable_pct",
        "finetuned_scope",
        "source_run",
        "notes",
    ]
    scored = [row for row in rows if num(row.get("tuned_score")) is not None]
    scored.sort(key=sort_key)
    lines = [
        "# TextVQA Full Train / Full Eval Results",
        "",
        f"Rows with tuned full-eval scores: {len(scored)}",
        "",
        "|" + "|".join(display_cols) + "|",
        "|" + "|".join(["---"] * len(display_cols)) + "|",
    ]
    for idx, row in enumerate(scored, 1):
        values = {
            "rank": idx,
            **row,
        }
        line = []
        for col in display_cols:
            text = str(values.get(col, ""))
            text = text.replace("|", "/").replace("\n", " ")
            line.append(text)
        lines.append("|" + "|".join(line) + "|")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    if LOCAL_QWEN.exists():
        for raw in read_csv(LOCAL_QWEN):
            row = normalize_local_qwen(raw)
            if row:
                rows.append(row)

    for source_run, path in REMOTE_SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        for raw in read_csv(path):
            row = normalize_remote(source_run, raw)
            if row:
                rows.append(row)

    rows.sort(key=sort_key)
    write_csv(OUT_DIR / "textvqa_full_train_eval_results_combined.csv", rows)
    write_markdown(OUT_DIR / "textvqa_full_train_eval_results_ranked.md", rows)
    print(f"wrote {len(rows)} rows")
    print(f"scored rows: {sum(1 for row in rows if num(row.get('tuned_score')) is not None)}")
    for row in rows[:10]:
        print(row["method"], row["tuned_score"], row["source_run"])


if __name__ == "__main__":
    main()
