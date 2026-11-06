"""QLoRA fine-tuning with prompt-masked loss (only the answer tokens are trained on).

Uses the plain transformers Trainer + peft (no TRL) so it does not depend on TRL's changing API.
Run one process per GPU:  CUDA_VISIBLE_DEVICES=0 python train_qlora.py --train t.jsonl --dev d.jsonl --out dir

Safety nets for long Kaggle runs: a wall-clock limit (--train_hours) that saves the adapter and stops cleanly,
a NaN guard, checkpoints every --save_steps with automatic resume, and a tokens/sec estimate for planning.
--tiny / --no_quant exist only for CPU smoke tests. Pure ASCII source on purpose.
"""
import argparse
import inspect
import json
import math
import os
import random
import time

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainerCallback, TrainingArguments)
from peft import LoraConfig, get_peft_model

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_len", type=int, default=2560)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--train_hours", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument("--no_quant", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--ddp_backend", default="", help="CPU smoke tests only (gloo); GPUs use NCCL automatically")
    return p.parse_args()


def encode_example(tok, ex, max_len, im_end_id):
    """Chat-format the prompt, append the answer, and mask the prompt in the labels. None if the prompt leaves no room."""
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt = tok(text, add_special_tokens=False)["input_ids"]
    ans = tok(ex["answer"], add_special_tokens=False)["input_ids"]
    room = max_len - len(prompt)
    if room < 256:
        return None
    if len(ans) + 1 > room:
        ans = ans[:room]  # truncated: no end token, so the model is never taught to stop mid-sentence
        truncated = True
    else:
        ans = ans + [im_end_id]
        truncated = False
    return {"input_ids": prompt + ans, "labels": [-100] * len(prompt) + ans, "truncated": truncated,
            "n_prompt": len(prompt)}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {"input_ids": self.items[i]["input_ids"], "labels": self.items[i]["labels"]}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), m), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), m), -100, dtype=torch.long)
        att = torch.zeros((len(feats), m), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def _dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class Guard(TrainerCallback):
    """Wall-clock limit + NaN guard. On a stop, a checkpoint is saved (so the next session resumes) and the adapter is
    written. With several processes (DDP) the stop decision is all-reduced, so every rank stops on the same step."""

    def __init__(self, hours):
        self.t_end = time.time() + hours * 3600
        self.reason = "completed"
        self.nan = False
        self.start_step = 0

    def on_train_begin(self, args, state, control, **kw):
        self.start_step = state.global_step      # > 0 when resumed from a checkpoint

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs and (math.isnan(logs["loss"]) or math.isinf(logs["loss"])):
            self.nan = True
        return control

    def on_step_end(self, args, state, control, **kw):
        flag = 2 if self.nan else (1 if time.time() > self.t_end else 0)
        if _dist():
            t = torch.tensor([flag], device="cuda" if torch.cuda.is_available() else "cpu")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            flag = int(t.item())
        if flag:
            self.reason = "nan_loss" if flag == 2 else "time_limit"
            if flag == 2:
                print("NaN/inf loss - stopping. Try a smaller model or lower lr.", flush=True)
            else:
                control.should_save = True               # checkpoint for the next session
            control.should_training_stop = True
        return control


def build_model(a, tok):
    cuda = torch.cuda.is_available()
    if a.tiny:
        cfg = AutoConfig.for_model("qwen2", vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=4096, tie_word_embeddings=True)
        return AutoModelForCausalLM.from_config(cfg)
    if a.no_quant:
        return AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if cuda else torch.float32)
    from peft import prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    rank = int(os.environ.get("LOCAL_RANK", 0))       # one full 4-bit copy per GPU under DDP
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, torch_dtype=torch.float16,
                                                 device_map={"": rank}, attn_implementation="sdpa")
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=not a.no_grad_ckpt,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})


def main():
    a = parse()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end_id is not None and im_end_id != tok.unk_token_id, "tokenizer has no <|im_end|> (use a Qwen instruct model)"

    t0 = time.time()
    train_raw = load_jsonl(a.train)
    enc = [e for e in (encode_example(tok, ex, a.max_len, im_end_id) for ex in train_raw) if e]
    avg_len = sum(len(e["input_ids"]) for e in enc) / max(1, len(enc))
    print("train examples: %d kept of %d | truncated answers: %d | avg tokens %.0f | max %d | tokenised in %.0fs" % (
        len(enc), len(train_raw), sum(e["truncated"] for e in enc), avg_len, max(len(e["input_ids"]) for e in enc),
        time.time() - t0), flush=True)
    dev_ds = None
    if a.dev:
        dev_enc = [e for e in (encode_example(tok, ex, a.max_len, im_end_id) for ex in load_jsonl(a.dev)) if e]
        dev_ds = ListDataset(dev_enc)
        print("dev examples:", len(dev_enc), flush=True)

    model = build_model(a, tok)
    model.config.use_cache = False
    if a.no_quant and not a.tiny and not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():  # fp16 training needs fp32 trainable weights (GradScaler cannot unscale fp16 grads)
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    sig = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=1,
              gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
              num_train_epochs=a.epochs, max_steps=a.max_steps, logging_steps=5,
              save_steps=a.save_steps, save_total_limit=2, fp16=cuda, optim=a.optim, report_to="none",
              remove_unused_columns=False, dataloader_num_workers=0, seed=a.seed, max_grad_norm=1.0,
              disable_tqdm=True, gradient_checkpointing=(cuda and not a.no_grad_ckpt and not a.no_quant))
    if dev_ds is not None:
        kw["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "steps"
        kw["eval_steps"] = a.eval_steps
    if "gradient_checkpointing_kwargs" in sig:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        kw["ddp_find_unused_parameters"] = False
    if a.ddp_backend:
        kw["ddp_backend"] = a.ddp_backend
    # 3% warm-up: old transformers has warmup_ratio; new transformers takes a float in warmup_steps as the ratio
    kw["warmup_ratio" if "warmup_ratio" in sig else "warmup_steps"] = 0.03
    dropped = sorted(k for k in kw if k not in sig)
    if dropped:
        print("WARNING: this transformers version has no TrainingArguments option(s):", dropped, "- ignoring them", flush=True)
        kw = {k: v for k, v in kw.items() if k in sig}
    guard = Guard(a.train_hours)
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=ListDataset(enc), eval_dataset=dev_ds,
                      data_collator=Collator(tok.pad_token_id), callbacks=[guard])
    resume = any(d.startswith("checkpoint-") for d in os.listdir(a.out))
    t1 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    elapsed = time.time() - t1
    final = os.path.join(a.out, "final")
    trainer.save_model(final)                      # main process only (Trainer handles it)
    if not trainer.is_world_process_zero():
        return
    tok.save_pretrained(final)
    steps = trainer.state.global_step
    run_steps = steps - guard.start_step
    per_step = elapsed / max(1, run_steps)
    ex_per_step = a.bs * a.grad_accum * world
    summary = {"status": guard.reason, "steps": steps, "steps_this_session": run_steps, "resumed_from_step": guard.start_step,
               "world_size": world, "elapsed_s": round(elapsed, 1), "sec_per_step": round(per_step, 2),
               "examples_per_step": ex_per_step, "avg_tokens": round(avg_len),
               "tokens_per_sec_est": round(avg_len * ex_per_step / max(1e-9, per_step)),
               "n_train": len(enc), "truncated_answers": sum(e["truncated"] for e in enc),
               "steps_per_epoch": len(enc) // ex_per_step, "max_len": a.max_len,
               "loss_log": [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h],
               "eval_log": [(h["step"], round(h["eval_loss"], 4)) for h in trainer.state.log_history if "eval_loss" in h]}
    json.dump(summary, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    print("DONE", {k: v for k, v in summary.items() if k not in ("loss_log", "eval_log")}, flush=True)


if __name__ == "__main__":
    main()
