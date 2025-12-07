from mlx_lm import load, generate


def load_formal_style_model():
    # 注意：adapter_path 这个参数名可能会因 mlx-lm 版本略有出入，
    # 如果报错，可以改用命令行 generate + subprocess 调用。
    model, tokenizer = load(
        "Qwen/Qwen2-7B-Instruct-MLX",
        tokenizer_config={"eos_token": "<|im_end|>"},
        adapter_path="/Users/zhexu/PycharmProjects/AutoScriptAgent/style-transfer-mlx/adapters/cams_formal",
    )
    return model, tokenizer


def rewrite_to_formal(model, tokenizer, text: str, max_tokens: int = 256) -> str:
    prompt = (
        "你是一位擅长正式书面语写作的中文编辑。\n"
        "任务：在尽量保持原文语义不变的前提下，把原文改写成正式、规范、书面化的风格。\n\n"
        f"原文：{text}\n\n"
        "改写："
    )
    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        # temperature=0.7,
        # top_p=0.9,
    )
    return out


def main():
    model, tokenizer = load_formal_style_model()
    src = "Is tomorrow okay? Baby [bitter] I'm just in a meeting. I'll send a message to the boss later and then I'll go to the workout + the grocery store."
    dest = rewrite_to_formal(model, tokenizer, src, 256)
    print("原文：", src)
    print("改写：", dest)


if __name__ == "__main__":
    main()
