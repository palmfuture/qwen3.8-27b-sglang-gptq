#!/usr/bin/env python3
"""Benchmark OpenAI-compatible LLM server (vLLM or SGLang) with streaming.
Matches vLLM prod settings: enable_thinking=false, temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5"""
import argparse, json, time, urllib.request

def stream_chat(base, model, prompt, max_tokens=256):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    tokens = 0
    resp = urllib.request.urlopen(req, timeout=600)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if "content" in delta and delta["content"]:
            if ttft is None:
                ttft = time.time() - t0
            tokens += 1
    total = time.time() - t0
    gen = total - (ttft or 0)
    return tokens, total, ttft, gen

PROMPTS = {
    "essay_400": "Write a detailed 400-word essay about the history of deep learning in neural networks.",
    "code_200": "Write a complete Python function that implements merge sort with detailed comments. Return only the code.",
    "think_200": "Explain step by step why the sky is blue. Think carefully.",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="Qwen3.8-27B-GPTQ")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    print(f"bench {args.base} model={args.model} max_tokens={args.max_tokens} n={args.n} (thinking off)")
    for name, prompt in PROMPTS.items():
        ts = []
        for i in range(args.n):
            try:
                tok, total, ttft, gen = stream_chat(args.base, args.model, prompt, args.max_tokens)
                spd = tok / gen if gen and gen > 0 else 0
                ts.append(spd)
                print(f"  {name} run{i+1}: tokens={tok} ttft={ttft:.2f}s total={total:.2f}s decode={spd:.1f} tok/s")
            except Exception as e:
                print(f"  {name} run{i+1}: ERROR {e}")
        if ts:
            print(f"  {name} avg decode: {sum(ts)/len(ts):.1f} tok/s")
