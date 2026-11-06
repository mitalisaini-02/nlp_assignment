"""Phase 5 generation worker: one process per GPU.

Reads a jobs file (jsonl: uid, adapter, prio, system, user), generates greedy answers in batches, and appends every
finished batch to --out immediately, so a crashed or time-limited run resumes where it stopped (done uids are skipped).
adapter is "nn" / "none" (a LoRA folder under --adapter_dir) or "base" (the same model with the adapter switched off:
the zero-shot baseline B1).  Pure ASCII source on purpose.

    CUDA_VISIBLE_DEVICES=0 python infer.py --jobs jobs.jsonl --adapter_dir /path --adapters nn --out gen_gpu0.jsonl
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
from peft import PeftModel


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--adapters", required=True, help="comma list handled by this worker, e.g. none,base")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_new", type=int, default=768)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--rep_penalty", type=float, default=1.05, help="HF repetition penalty (acts on prompt AND answer tokens)")
    p.add_argument("--gen_rep_penalty", type=float, default=1.0, help="penalty on tokens of the ANSWER only (1.0 = off)")
    p.add_argument("--gen_no_repeat", type=int, default=0, help="forbid repeating any n-gram of the ANSWER (0 = off)")
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--no_quant", action="store_true")
    return p.parse_args()


class GenRepetitionPenalty(LogitsProcessor):
    """Repetition penalty over the generated tokens only (the HF one also penalises every prompt token, which
    discourages copying wording from the reference answers in the prompt). Assumes left padding."""

    def __init__(self, penalty, prompt_len):
        self.p, self.plen = penalty, prompt_len

    def __call__(self, input_ids, scores):
        g = input_ids[:, self.plen:]
        if g.shape[1] == 0:
            return scores
        s = torch.gather(scores, 1, g)
        s = torch.where(s < 0, s * self.p, s / self.p)
        return scores.scatter(1, g, s)


class GenNoRepeatNGram(LogitsProcessor):
    """Forbid any n-gram that already occurs in the generated answer (prompt excluded). Incremental per row."""

    def __init__(self, n, prompt_len):
        self.n, self.plen, self.seen = n, prompt_len, None

    def __call__(self, input_ids, scores):
        B, L = input_ids.shape
        if self.seen is None:
            self.seen = [dict() for _ in range(B)]
        glen = L - self.plen
        if glen < self.n - 1:
            return scores
        rows = input_ids[:, self.plen:].tolist()
        for b in range(B):
            g = rows[b]
            if glen >= self.n:                      # register the newest complete n-gram
                self.seen[b].setdefault(tuple(g[-self.n:-1]), set()).add(g[-1])
            banned = self.seen[b].get(tuple(g[len(g) - self.n + 1:]))
            if banned:
                scores[b, list(banned)] = -float("inf")
        return scores


def load_model(a, names):
    cuda = torch.cuda.is_available()
    if a.no_quant:
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if cuda else torch.float32)
        if cuda:
            base = base.to("cuda:0")
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, torch_dtype=torch.float16,
                                                    device_map={"": 0})
    real = [n for n in names if n != "base"]
    if not real:
        return base.eval(), False
    model = PeftModel.from_pretrained(base, os.path.join(a.adapter_dir, real[0]), adapter_name=real[0])
    for n in real[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, n), adapter_name=n)
    return model.eval(), True


def generate(model, tok, texts, a, device):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    plen = enc["input_ids"].shape[1]
    procs = LogitsProcessorList()
    if getattr(a, "gen_rep_penalty", 1.0) != 1.0:
        procs.append(GenRepetitionPenalty(a.gen_rep_penalty, plen))
    if getattr(a, "gen_no_repeat", 0):
        procs.append(GenNoRepeatNGram(a.gen_no_repeat, plen))
    kw = dict(max_new_tokens=a.max_new, repetition_penalty=a.rep_penalty, pad_token_id=tok.pad_token_id,
              logits_processor=procs)
    if getattr(a, "do_sample", False):
        torch.manual_seed(a.seed)
        kw.update(do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=0)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    res = []
    for g in gen:
        ids = g.tolist()
        n = len(ids)
        while n and ids[n - 1] == tok.pad_token_id:
            n -= 1
        res.append((tok.decode(ids[:n], skip_special_tokens=True).strip(), n))
    return res


def main():
    a = parse()
    names = a.adapters.split(",")
    jobs = [json.loads(l) for l in open(a.jobs, encoding="utf-8")]
    jobs = [j for j in jobs if j["adapter"] in names]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["uid"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [j for j in jobs if j["uid"] not in done]
    print("jobs for %s: %d | already done: %d | to do: %d" % (names, len(jobs), len(done), len(todo)), flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for j in todo:
        j["text"] = tok.apply_chat_template([{"role": "system", "content": j["system"]}, {"role": "user", "content": j["user"]}],
                                            tokenize=False, add_generation_prompt=True)
        j["plen"] = len(tok(j["text"], add_special_tokens=False)["input_ids"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, has_peft = load_model(a, names)
    t_end = time.time() + a.hours * 3600
    t0 = time.time()
    n_done = 0
    fout = open(a.out, "a", encoding="utf-8")
    for name in names:
        group = sorted((j for j in todo if j["adapter"] == name), key=lambda j: (j["prio"], -j["plen"]))
        if not group:
            continue
        if name != "base" and has_peft:
            model.set_adapter(name)
        print("adapter %s: %d jobs" % (name, len(group)), flush=True)
        i = 0
        bs = a.batch
        while i < len(group):
            if time.time() > t_end:
                print("time limit reached - stopping; rerun to resume", flush=True)
                return
            batch = group[i:i + bs]
            try:
                if name == "base" and has_peft:
                    with model.disable_adapter():
                        res = generate(model, tok, [j["text"] for j in batch], a, device)
                else:
                    res = generate(model, tok, [j["text"] for j in batch], a, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print("out of memory - batch size now", bs, flush=True)
                continue
            for j, (text, n) in zip(batch, res):
                fout.write(json.dumps({"uid": j["uid"], "adapter": name, "text": text, "n_tokens": n,
                                       "hit_limit": n >= a.max_new}, ensure_ascii=False) + "\n")
            fout.flush()
            i += len(batch)
            n_done += len(batch)
            el = time.time() - t0
            print("[%5.0f min] %s %d/%d | %.1f s per answer" % (el / 60, name, i, len(group), el / n_done), flush=True)
    print("DONE", n_done, "answers in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
