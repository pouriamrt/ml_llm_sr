import os
from config import ensure_dirs, Paths

def sh(cmd):
    print(f"==> {cmd}")
    assert os.system(cmd) == 0, f"Command failed: {cmd}"

if __name__ == "__main__":
    paths = Paths()
    ensure_dirs(paths)

    # 1) Embed
    sh("python data_prep.py")
    # 2) Train supervised
    sh("python train_supervised.py")
    # 3) LLM weak labels (needs OPENAI_API_KEY in env)
    sh("python llm_label.py")
    for i in range(3):
        # 4) Fuse + predict
        sh("python fuse_and_predict.py")
        # 5) (Optional) Self-training
        sh("python self_training.py")
