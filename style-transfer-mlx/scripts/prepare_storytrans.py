# scripts/prepare_storytrans.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
STORYTRANS_DIR = ROOT / "raw_datasets" / "storytrans_public" / "text_style_transfer" / "data" / "zh"


STYLE_CONFIG = {
    "lx": {
        "name": "鲁迅",
        "src_file": STORYTRANS_DIR / "LX" / "train.json",
    },
    "jy": {
        "name": "金庸",
        "src_file": STORYTRANS_DIR / "JY" / "train.json",
    },
    "tale": {
        "name": "童话",
        "src_file": STORYTRANS_DIR / "tale" / "train.json",
    },
}


def load_story_list(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    # StoryTrans 中文这边通常是 list[{"text": "..."}] 的结构，具体可 print 一下确认
    return data


def build_instruct_sample(style_name: str, story: str) -> str:
    return (
        f"指令：仿照{style_name}的写作风格，写一小段中文故事。\n"
        f"输出：{story.strip()}"
    )


def prepare_style(style_key: str, cfg: Dict):
    style_name = cfg["name"]
    src_file: Path = cfg["src_file"]
    out_dir = ROOT / "data" / f"storytrans_{style_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stories = load_story_list(src_file)

    out_path = out_dir / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(stories, desc=f"storytrans-{style_key}"):
            text = ex.get("text", "").replace("\r", " ").strip()
            if not text:
                continue
            sample = {"text": build_instruct_sample(style_name, text)}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[OK] {style_key}: {len(stories)} samples -> {out_path}")


def main():
    for k, v in STYLE_CONFIG.items():
        prepare_style(k, v)


if __name__ == "__main__":
    main()
