from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm


def build_formal_dataset(
    out_dir: Path,
    n_train: int = 50000,
    n_valid: int = 1000,
    n_test: int = 1000,
):
    print("Loading CAMS dataset...")
    ds = load_dataset("Mxode/CAMS", split="train")  # 100 万条左右 :contentReference[oaicite:13]{index=13}

    def is_formal(example):
        formality = example.get("formality", "")
        return "正式" in formality

    formal = ds.filter(is_formal)
    print("Total formal samples:", len(formal))

    formal = formal.shuffle(seed=42)

    n_train = min(n_train, len(formal))
    n_valid = min(n_valid, max(0, len(formal) - n_train))
    n_test = min(n_test, max(0, len(formal) - n_train - n_valid))

    train = formal.select(range(0, n_train))
    valid = formal.select(range(n_train, n_train + n_valid))
    test = formal.select(range(n_train + n_valid, n_train + n_valid + n_test))

    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(split, name: str):
        path = out_dir / f"{name}.jsonl"
        print(f"Writing {name} -> {path}")
        with path.open("w", encoding="utf-8") as f:
            for ex in tqdm(split, desc=name):
                formality = ex.get("formality", "")
                text = ex.get("text", "").replace("\n", " ").strip()
                # 简单把风格标签写进前缀，方便以后 prompt 控制
                cn_style = "正式" if "正式" in str(formality) else "口语"
                train_text = (
                    f"【风格：{cn_style}】\n"
                    f"{text}"
                )
                json.dump({"text": train_text}, f, ensure_ascii=False)
                f.write("\n")

    dump(train, "train")
    dump(valid, "valid")
    dump(test, "test")


def main():
    out_dir = Path("data/cams_formal")
    build_formal_dataset(out_dir)


if __name__ == "__main__":
    main()
