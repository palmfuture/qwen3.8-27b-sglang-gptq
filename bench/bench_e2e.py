#!/usr/bin/env python3
"""Non-streaming benchmark — measures wall-clock E2E throughput exactly like a client sees it.
No SSE parsing, no TTFT subtraction. Uses usage.completion_tokens for the authoritative token count."""
import argparse, json, time, urllib.request

def chat(base, model, prompt, max_tokens=512, greedy=True):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if greedy:
        body.update(temperature=0.0, top_p=1.0)
    else:
        body.update(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
    req = urllib.request.Request(base + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=600)
    d = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    usage = d.get("usage", {})
    comp = usage.get("completion_tokens", 0)
    return comp, elapsed, d.get("choices",[{}])[0].get("finish_reason")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="Qwen3.8-27B-GPTQ")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-greedy", dest="greedy", action="store_false", default=True)
    args = ap.parse_args()
    print(f"non-stream bench {args.base} model={args.model} n={args.n} greedy={args.greedy}")
    for name, prompt in {
        "essay_400": "Write a detailed 400-word essay about the history of deep learning.",
        "code_200": "Write a complete Python function that implements merge sort with detailed comments.",
    }.items():
        rates = []
        for i in range(args.n):
            try:
                comp, elapsed, fin = chat(args.base, args.model, prompt, args.max_tokens, args.greedy)
                rate = comp/elapsed if elapsed>0 else 0
                rates.append(rate)
                print(f"  {name} run{i+1}: tokens={comp} elapsed={elapsed:.2f}s rate={rate:.1f} tok/s finish={fin}")
            except Exception as e:
                print(f"  {name} run{i+1}: ERROR {e}")
        if rates:
            print(f"  {name} avg: {sum(rates)/len(rates):.1f} tok/s")
