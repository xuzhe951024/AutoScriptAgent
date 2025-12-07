# scripts/infer_cli.py
from __future__ import annotations

import textwrap
from pathlib import Path

from  cli_wrapper import MLXLMCliConfig, MLXLMCliWrapper


def build_formal_rewrite_prompt(text: str) -> str:
    """
    构造一个喂给 CLI 的 prompt。

    这里不走 chat_template，而是直接把“角色设定 + 任务描述 + 原文”
    组织成一个长 prompt，行为会和你在 shell 里手动敲 prompt 一致。
    """
    prompt = textwrap.dedent(f"""
    你是一位擅长正式书面语写作的中文编辑。
    请在尽量保持原文语义不变的前提下，把下面这段话改写成正式、规范、书面化的中文风格。

    原文：{text}

    改写：
    """).strip()
    return prompt


def main():
    # 假定你在项目根目录下用 `uv run python scripts/infer_cli.py` 运行
    project_root = Path(__file__).resolve().parents[1]
    adapters_dir = project_root / "adapters" / "cams_formal"

    cfg = MLXLMCliConfig(
        model="Qwen/Qwen2-7B-Instruct-MLX",
        adapter_path=str(adapters_dir),
        max_tokens=256,
        temp=0.1,   # 尽量模拟 CLI 默认采样温度
        top_p=0.1,
        seed=None,     # 固定随机性，方便复现
        extra_args=[
            # 如果你不想用 chat 模板，可以加上：
            # "--ignore-chat-template",
            # 反之，如果你要启用某些特定行为，也可以在这里追加。
        ],
    )

    wrapper = MLXLMCliWrapper(cfg)

    src = "Is tomorrow okay? Baby [bitter] I'm just in a meeting. I'll send a message to the boss later and then I'll go to the workout + the grocery store."
    prompt = build_formal_rewrite_prompt(src)

    result = wrapper.generate(prompt)

    print("原文：")
    print(src)
    print("\n改写：")
    print(result)


if __name__ == "__main__":
    main()
