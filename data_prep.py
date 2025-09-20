import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from config import Paths, ModelCfg, ensure_dirs
from utils import prep_text_df, save_numpy

def main():
    paths = Paths()
    cfg = ModelCfg()
    ensure_dirs(paths)
    
    df_include = pd.read_csv(paths.labeled_include_csv)
    df_include['label'] = 'include'
    df_exclude = pd.read_csv(paths.labeled_exclude_csv)
    df_exclude['label'] = 'exclude'
    if os.path.exists(paths.labeled_maybe_csv):
        df_maybe = pd.read_csv(paths.labeled_maybe_csv)
        df_maybe['label'] = 'maybe'
        gold = pd.concat([df_include, df_exclude, df_maybe])
    else:
        gold = pd.concat([df_include, df_exclude])
    gold = gold.dropna(subset=["Title", "Abstract Note"])
    gold = gold[cfg.columns + ["label"]]
    unl  = pd.read_csv(paths.unlabeled_csv)
    unl = unl.drop_duplicates(subset=["Key"]).dropna(subset=["Title", "Abstract Note"])
    unl = unl[cfg.columns]

    gold = prep_text_df(gold)
    unl  = prep_text_df(unl)

    if os.path.exists(os.path.join(paths.emb_dir, "gold.npy")) and os.path.exists(os.path.join(paths.emb_dir, "unl.npy")):
        print("Gold and unlabeled embeddings already exist, do you want to overwrite them? (y/n): ", end="")
        overwrite = input()
        while overwrite.lower() not in ["y", "n"]:
            print("\n Invalid input, please enter y or n: ", end="")
            overwrite = input()

    if overwrite.lower() == "y":
        model = SentenceTransformer(cfg.embed_model)

        print("Embedding gold...")
        Xg = model.encode(gold["text"].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True)
        print("Embedding unlabeled...")
        Xu = model.encode(unl["text"].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True)

        save_numpy(os.path.join(paths.emb_dir, "gold.npy"), Xg)
        save_numpy(os.path.join(paths.emb_dir, "unl.npy"), Xu)

    gold.to_parquet(os.path.join(paths.emb_dir, "gold.parquet"))
    unl.to_parquet(os.path.join(paths.emb_dir, "unl.parquet"))
    print("Saved embeddings.")

if __name__ == "__main__":
    main()
