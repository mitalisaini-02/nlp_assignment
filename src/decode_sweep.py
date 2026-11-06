"""Decoding sweep worker: one process per GPU. Loads the base model + one adapter once, then answers the same prompts
under several decoding configurations. Appends every batch to --out (resumable: done uids are skipped).

    CUDA_VISIBLE_DEVICES=0 python decode_sweep.py --payload payload.json --configs A,C --adapter_dir /path --out g0.jsonl
Pure ASCII source on purpose.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import infer

# name: decoding settings. max_new 1280 for all (the 768 cap cut ~40% of references short).
BASE = dict(max_new=1280, rep_penalty=1.0, gen_rep_penalty=1.0, gen_no_repeat=0, do_sample=False, temperature=0.7, top_p=0.9)
CONFIGS = {
    "A_greedy_rp1.05": dict(rep_penalty=1.05),                                    # Phase 5 decoding, higher cap only
    "B_greedy_rp1.15_hf": dict(rep_penalty=1.15),                                  # standard HF penalty (also hits prompt words)
    "C_greedy_genrp1.15": dict(gen_rep_penalty=1.15),                              # penalty on the answer's own tokens only
    "D_greedy_rp1.05_nr12": dict(rep_penalty=1.05, gen_no_repeat=12),              # no repeated 12-token n-gram in the answer
    "E_greedy_genrp1.1_nr12": dict(gen_rep_penalty=1.1, gen_no_repeat=12),
    "F_sample_t0.7_rp1.05": dict(rep_penalty=1.05, do_sample=True),
    "G_sample_t0.7_genrp1.1_nr12": dict(gen_rep_penalty=1.1, gen_no_repeat=12, do_sample=True),
}


def settings(name, max_new=None):
    s = dict(BASE, **CONFIGS[name])
    if max_new:
        s["max_new"] = max_new
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--payload", required=True)
    p.add_argument("--configs", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--adapter", default="nn")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--max_new", type=int, default=0, help="override max_new (smoke tests)")
    p.add_argument("--no_quant", action="store_true")
    a = p.parse_args()

    items = json.load(open(a.payload, encoding="utf-8"))["items"]
    names = a.configs.split(",")
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["uid"] for l in open(a.out, encoding="utf-8") if l.strip()}
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for it in items:
        it["text"] = tok.apply_chat_template([{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}],
                                             tokenize=False, add_generation_prompt=True)
        it["plen"] = len(tok(it["text"], add_special_tokens=False)["input_ids"])
    items.sort(key=lambda it: -it["plen"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _ = infer.load_model(SimpleNamespace(no_quant=a.no_quant, model=a.model, adapter_dir=a.adapter_dir), [a.adapter])
    model.set_adapter(a.adapter)
    fout = open(a.out, "a", encoding="utf-8")
    for name in names:
        s = settings(name, a.max_new or None)
        todo = [it for it in items if "%s|%s" % (name, it["qid"]) not in done]
        print("config %s: %s | to do %d" % (name, s, len(todo)), flush=True)
        t0, bs, i, bi = time.time(), a.batch, 0, 0
        while i < len(todo):
            batch = todo[i:i + bs]
            ga = SimpleNamespace(seed=1234 + bi, **s)
            tb = time.time()
            try:
                res = infer.generate(model, tok, [it["text"] for it in batch], ga, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print("out of memory - batch size now", bs, flush=True)
                continue
            sec = (time.time() - tb) / len(batch)
            for it, (text, n) in zip(batch, res):
                fout.write(json.dumps({"uid": "%s|%s" % (name, it["qid"]), "config": name, "qid": it["qid"], "text": text,
                                       "n_tokens": n, "hit_limit": n >= s["max_new"], "sec_per_answer": sec}, ensure_ascii=False) + "\n")
            fout.flush()
            i += len(batch)
            bi += 1
            print("[%5.1f min] %s %d/%d | %.1f s per answer" % ((time.time() - t0) / 60, name, i, len(todo), sec), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
