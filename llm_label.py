# llm_label_async.py
import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from openai import AsyncOpenAI
from config import Paths, LLMCfg, ensure_dirs
from prompts import SYSTEM_PROMPT, USER_INSTRUCTION, FEWSHOTS

# ---------- helpers ----------

def confidence_to_prob(label: str, conf: str, mapping: Dict[str, float]) -> float:
    label = (label or "maybe").strip().lower()
    conf  = (conf  or "low").strip().lower()
    if label == "include":
        return mapping.get(conf, 0.7)
    if label == "maybe":
        return 0.5
    # exclude
    return 1.0 - mapping.get(conf, 0.7)

def build_messages(text: str, use_fewshots: bool = False) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_fewshots and FEWSHOTS:
        messages.extend(FEWSHOTS)
    messages.append({"role": "user", "content": USER_INSTRUCTION.format(TEXT=text)})
    return messages

async def call_openai_one(
    client: AsyncOpenAI,
    model: str,
    text: str,
    temperature: float,
    max_retries: int = 3,
    use_fewshots: bool = False,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Call OpenAI once with retries/backoff. Always returns a dict with the expected keys."""
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            messages = build_messages(text, use_fewshots=use_fewshots)
            # timeout at the task level
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout_s,
            )
            payload = (resp.choices[0].message.content or "").strip()
            obj = json.loads(payload)
            # minimal sanity
            return {
                "llm_label": (obj.get("label") or "maybe").lower().strip(),
                "llm_conf": (obj.get("confidence") or "low").lower().strip(),
                "llm_reasons": obj.get("reasons", ""),
                "llm_evidence_spans": obj.get("evidence_spans", []),
            }
        except Exception:
            if attempt + 1 >= max_retries:
                return {
                    "llm_label": "maybe",
                    "llm_conf": "low",
                    "llm_reasons": "parse_error",
                    "llm_evidence_spans": [],
                }
            await asyncio.sleep(backoff * (attempt + 1))

async def process_split(
    client: AsyncOpenAI,
    split: str,
    cfg: LLMCfg,
    paths: Paths,
    concurrency: Optional[int] = None,
):
    df = pd.read_parquet(os.path.join(paths.emb_dir, f"{split}.parquet"))
    texts = df["text"].tolist()

    sem = asyncio.Semaphore(concurrency or getattr(cfg, "concurrency", 16))
    results: List[Optional[Dict[str, Any]]] = [None] * len(texts)

    async def worker(idx: int, txt: str):
        async with sem:
            res = await call_openai_one(
                client=client,
                model=cfg.model,
                text=txt,
                temperature=cfg.temperature,
                max_retries=cfg.max_retries,
                use_fewshots=getattr(cfg, "use_fewshots", False),
                timeout_s=getattr(cfg, "timeout_s", 60.0),
            )
            # add the derived probability
            p_inc = confidence_to_prob(res["llm_label"], res["llm_conf"], cfg.confidence_map)
            res.update({"idx": idx, "llm_prob_include": p_inc})
            results[idx] = res

    tasks = [asyncio.create_task(worker(i, t)) for i, t in enumerate(texts)]

    # progress bar with as_completed
    with tqdm(total=len(tasks), desc=f"LLM {split}") as pbar:
        for fut in asyncio.as_completed(tasks):
            await fut
            pbar.update(1)

    # materialize rows in original order
    rows = [r for r in results if r is not None]
    out_df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    out_path = os.path.join(paths.llm_dir, f"llm_labels_{split}.parquet")
    os.makedirs(paths.llm_dir, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"Wrote: {out_path}")

# ---------- entrypoint ----------

async def amain():
    # Windows event loop fix
    if sys.platform.startswith("win"):
        try:
            import asyncio.windows_events
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    paths = Paths()
    cfg = LLMCfg()
    ensure_dirs(paths)
    
    if getattr(cfg, "api_key", None):
        client = AsyncOpenAI(api_key=cfg.api_key)
    else:
        client = AsyncOpenAI()

    # run splits sequentially
    for split in ["gold", "unl"]:
        if os.path.exists(os.path.join(paths.llm_dir, f"llm_labels_{split}.parquet")):
            print(f"LLM labels for {split} already exist, Do you want to overwrite them? (y/n): ", end="")
            overwrite = input()
            while overwrite.lower() not in ["y", "n"]:
                print("\n Invalid input, please enter y or n: ", end="")
                overwrite = input()
            if overwrite.lower() == "n":
                continue
        await process_split(client, split, cfg, paths, concurrency=cfg.concurrency)

def main():
    asyncio.run(amain())

if __name__ == "__main__":
    main()
