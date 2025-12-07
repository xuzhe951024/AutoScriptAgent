# style_transfer/cli_wrapper.py
from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MLXLMCliConfig:
    """
    配置 mlx_lm.generate 的参数集合。

    这个 dataclass 只负责描述“想让 CLI 怎么跑”，不耦合具体任务。
    """
    model: str = "Qwen/Qwen2-7B-Instruct-MLX"
    adapter_path: Optional[str] = None
    max_tokens: int = 256
    temp: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    seed: Optional[int] = 0

    # 用哪个 Python 调 CLI，默认就是当前这个 venv 里的解释器
    python_executable: str = sys.executable

    # 额外 CLI 参数（比如 --ignore-chat-template / --verbose 等）
    extra_args: List[str] = field(default_factory=list)

    # 是否在出错时打印命令（debug 好用）
    echo_cmd_on_error: bool = True


class MLXLMCliError(RuntimeError):
    """封装一下错误类型，方便上层精细处理。"""
    pass


class MLXLMCliWrapper:
    """
    通过 `python -m mlx_lm.generate` 调用模型的封装。

    设计目标：
    - 与 CLI 行为尽量一致（包括默认 temp / top-p）
    - 不和 mlx-lm 的 Python 内部 API 纠缠
    - 出错时给出比较友好的错误信息
    """

    def __init__(self, config: MLXLMCliConfig):
        self.config = config

    def _build_cmd(self, prompt: str) -> List[str]:
        cfg = self.config

        cmd: List[str] = [
            cfg.python_executable,
            "-m",
            "mlx_lm.generate",
            "--model",
            cfg.model,
            "--max-tokens",
            str(cfg.max_tokens),
        ]

        if cfg.adapter_path:
            cmd += ["--adapter-path", str(cfg.adapter_path)]

        if cfg.temp is not None:
            cmd += ["--temp", str(cfg.temp)]

        if cfg.top_p is not None:
            cmd += ["--top-p", str(cfg.top_p)]

        if cfg.seed is not None:
            cmd += ["--seed", str(cfg.seed)]

        # 额外参数直接拼上去
        if cfg.extra_args:
            cmd += cfg.extra_args

        # prompt 最后加，保证顺序和你手动敲命令一致
        cmd += ["--prompt", prompt]

        return cmd

    def generate(self, prompt: str, timeout: Optional[float] = None) -> str:
        """
        调用 mlx_lm.generate，返回 stdout 文本（去掉首尾空白）。

        如果命令返回非零 exit code，会抛 MLXLMCliError。
        """
        cmd = self._build_cmd(prompt)

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise MLXLMCliError(f"mlx_lm.generate 超时（timeout={timeout}s）") from e
        except OSError as e:
            # 比如 python 可执行文件不存在
            raise MLXLMCliError(f"执行命令失败：{e}") from e

        if proc.returncode != 0:
            msg_lines = [
                f"mlx_lm.generate 退出码 {proc.returncode}",
            ]
            if self.config.echo_cmd_on_error:
                cmd_str = " ".join(map(str, cmd))
                msg_lines.append(f"命令: {cmd_str}")
            if proc.stderr:
                msg_lines.append("stderr:")
                msg_lines.append(proc.stderr.strip())
            raise MLXLMCliError("\n".join(msg_lines))

        # 正常情况下，stdout 就是生成结果（可能有结尾换行）
        return proc.stdout.strip()
