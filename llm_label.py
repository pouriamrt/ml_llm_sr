import os
import time
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from openai import OpenAI
from config import Paths, LLMCfg, LABEL_MAP, ensure_dirs
from prompts import SYSTEM_PROMPT, USER_INSTRUCTION, FEWSHOTS

def call_openai_batch(client: OpenAI, model: str, batch_texts: List[str], temperature: float = 0.0) -> List[Dict[str, Any]]:
    out = []
    for txt in batch_texts:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # messages.extend(FEWSHOTS)
        messages.append({"role": "user", "content": USER_INSTRUCTION.format(TEXT=txt)})
        resp = client.chat.completions.create(
            model=model, temperature=temperature, messages=messages, response_format={"type":"json_object"}
        )
        try:
            payload = resp.choices[0].message.content
            obj = json.loads(payload)
        except Exception:
            obj = {"label":"maybe","confidence":"low","reasons":"parse_error","evidence_spans":[]}
        out.append(obj)
    return out

def confidence_to_prob(label: str, conf: str, mapping: Dict[str, float]) -> float:
    # Return probability of 'include' derived from label + confidence.
    if label == "include":
        return mapping.get(conf, 0.7)
    if label == "maybe":
        return 0.5
    # exclude
    return 1.0 - mapping.get(conf, 0.7)

def main():
    paths = Paths()
    cfg = LLMCfg()
    ensure_dirs(paths)

    client = OpenAI(api_key=cfg.api_key)

    for split in ["gold","unl"]:
        df = pd.read_parquet(os.path.join(paths.emb_dir, f"{split}.parquet"))
        texts = df["text"].tolist()

        rows = []
        for i in tqdm(range(0, len(texts), cfg.batch_size), desc=f"LLM {split}"):
            batch = texts[i:i+cfg.batch_size]

            for attempt in range(cfg.max_retries):
                try:
                    preds = call_openai_batch(client, cfg.model, batch, cfg.temperature)
                    break
                except Exception as e:
                    if attempt+1 == cfg.max_retries:
                        raise
                    time.sleep(1.5*(attempt+1))

            for j, obj in enumerate(preds):
                lab = (obj.get("label") or "maybe").lower().strip()
                conf = (obj.get("confidence") or "low").lower().strip()
                reasons = obj.get("reasons","")
                spans = obj.get("evidence_spans",[])
                p_inc = confidence_to_prob(lab, conf, cfg.confidence_map)
                rows.append({
                    "idx": i+j,
                    "llm_label": lab,
                    "llm_conf": conf,
                    "llm_prob_include": p_inc,
                    "llm_reasons": reasons,
                    "llm_evidence_spans": spans
                })

        out_df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
        out_path = os.path.join(paths.llm_dir, f"llm_labels_{split}.parquet")
        out_df.to_parquet(out_path)
        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
