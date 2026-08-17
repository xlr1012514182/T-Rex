"""Actual Qwen3-VL Planner LoRA SFT entry point.

The ``--dry-run`` path performs the complete real-data audit without importing
torch/transformers/peft or loading a model.  A non-dry run trains only LoRA
adapters on the four aligned images and canonical planner JSON response.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .planner import SYSTEM_CONSTRAINTS
from .training import PlannerLoRAConfig, validate_planner_dataset
from .artifacts import (
    ADAPTER_MANIFEST_NAME,
    PLANNER_ADAPTER_SCHEMA,
    PROCESSOR_SCHEMA,
    adapter_files_sha256,
    processor_identity,
    processor_sha256,
)


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _user_prompt(row: Mapping[str, Any]) -> str:
    primitive = str(row["primitive"])
    timestamps = tuple(int(value) for value in row["frame_timestamps_ns"])
    ages_ms = [round((timestamps[-1] - value) / 1_000_000, 3) for value in timestamps[:3]]
    return (
        SYSTEM_CONSTRAINTS
        + f"\nLatched primitive: {primitive}. Copy it exactly."
        + "\nImage order: full(t-2), full(t-1), full(t), center(t)."
        + f" Full-frame ages_ms={ages_ms}."
    )


class _PlannerDataset:
    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.rows = _read_rows(root / f"{split}.jsonl")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


class _PlannerCollator:
    def __init__(self, processor: Any, root: Path, max_length: int) -> None:
        self.processor = processor
        self.root = root
        self.max_length = max_length

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        # V1 fixes per-device batch=1; effective batching is gradient accumulation.
        if len(examples) != 1:
            raise ValueError("Planner V1 collator requires per_device_train_batch_size=1")
        from PIL import Image

        row = examples[0]
        paths = tuple(row["full_view_paths"]) + (row["center_view_path"],)
        images = []
        try:
            for relative in paths:
                images.append(Image.open(self.root / str(relative)).convert("RGB"))
            user_content = [{"type": "image", "image": image} for image in images]
            user_content.append({"type": "text", "text": _user_prompt(row)})
            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = prompt_messages + [{
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": json.dumps(row["output"], ensure_ascii=False, separators=(",", ":")),
                }],
            }]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            encoded = self.processor(
                text=[full_text],
                images=[images],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_encoded = self.processor(
                text=[prompt_text],
                images=[images],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_length = int(prompt_encoded["attention_mask"][0].sum().item())
            labels = encoded["input_ids"].clone()
            labels[:, :prompt_length] = -100
            labels[encoded["attention_mask"] == 0] = -100
            encoded["labels"] = labels
            return encoded
        finally:
            for image in images:
                image.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = PlannerLoRAConfig()
    parser.add_argument("--model", default=defaults.base_model)
    parser.add_argument("--model-revision", default=defaults.base_revision)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=defaults.rank)
    parser.add_argument("--lora-alpha", type=int, default=defaults.alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.dropout)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--effective-batch", type=int, default=defaults.effective_batch)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--max-sequence-length", type=int, default=defaults.max_sequence_length)
    parser.add_argument("--precision", default=defaults.precision)
    parser.add_argument("--target-modules", default=",".join(defaults.target_modules))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PlannerLoRAConfig(
        base_model=args.model,
        base_revision=args.model_revision,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        effective_batch=args.effective_batch,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_sequence_length=args.max_sequence_length,
        precision=args.precision,
        target_modules=tuple(value.strip() for value in args.target_modules.split(",") if value.strip()),
    )
    config.validate()
    audit = validate_planner_dataset(args.dataset)
    run_contract = {
        "schema_version": "planner-lora-run-v1",
        "config": config.__dict__,
        "dataset": audit.__dict__,
        "model_revision": args.model_revision,
        "processor_schema": PROCESSOR_SCHEMA,
        "dry_run": bool(args.dry_run),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_contract.json").write_text(
        json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Planner LoRA training requires torch, transformers with Qwen3-VL, and peft"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Planner bf16 LoRA training requires a CUDA GPU")

    processor = AutoProcessor.from_pretrained(
        config.base_model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.base_model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    root = Path(audit.root)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.effective_batch,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            seed=args.seed,
            remove_unused_columns=False,
            report_to=[],
        ),
        train_dataset=_PlannerDataset(root, "train"),
        eval_dataset=_PlannerDataset(root, "val"),
        data_collator=_PlannerCollator(processor, root, config.max_sequence_length),
    )
    trainer.train()
    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    manifest = {
        "schema_version": PLANNER_ADAPTER_SCHEMA,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "processor_schema": PROCESSOR_SCHEMA,
        "processor_identity": processor_identity(processor),
        "processor_sha256": processor_sha256(processor),
        "adapter_files_sha256": adapter_files_sha256(adapter_dir),
        "dataset_audit": audit.__dict__,
        "lora_config": config.__dict__,
        "verification_scope": "real-camera Planner LoRA SFT artifact; no robot-success claim",
    }
    (adapter_dir / ADAPTER_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
