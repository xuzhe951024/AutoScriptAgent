#!/usr/bin/env bash
set -euo pipefail

# 计算项目根目录（脚本所在目录的上一级）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/raw_datasets"

mkdir -p "${DATA_ROOT}"

echo "项目根目录: ${ROOT_DIR}"
echo "数据根目录: ${DATA_ROOT}"
echo

############################################
# 1) 下载 StoryTrans 仓库（含中文作者风格语料）
############################################
STORY_DIR="${DATA_ROOT}/storytrans_public"

if [ -d "${STORY_DIR}/text_style_transfer/data/zh" ]; then
  echo "[StoryTrans] 已存在，无需重新 clone: ${STORY_DIR}"
else
  echo "[StoryTrans] 开始克隆仓库到 ${STORY_DIR} ..."
  git clone https://github.com/Xuekai-Zhu/storytrans_public.git "${STORY_DIR}"
  echo "[StoryTrans] 克隆完成。中文数据在:"
  echo "  ${STORY_DIR}/text_style_transfer/data/zh"
  echo "  # 其中包括：童话、鲁迅(LX)、金庸(JY) 等风格语料。"
fi

echo

########################################################
# 2) 下载 Ancient–Modern Chinese 平行语料（古今文）
########################################################
AM_RAW_TAR="${DATA_ROOT}/chinese_ancient_modern.tar"
AM_OUT_DIR="${DATA_ROOT}/ancient_modern"

if [ -d "${AM_OUT_DIR}" ] && [ -n "$(ls -A "${AM_OUT_DIR}" 2>/dev/null || true)" ]; then
  echo "[Ancient-Modern] 已存在解压目录: ${AM_OUT_DIR}"
else
  mkdir -p "${AM_OUT_DIR}"
  if [ -f "${AM_RAW_TAR}" ]; then
    echo "[Ancient-Modern] 已存在压缩包: ${AM_RAW_TAR}"
  else
    echo "[Ancient-Modern] 开始下载 chinese_ancient_modern.tar ..."

    # 优先使用 gdown（更稳处理 Google Drive 大文件）
    if command -v gdown >/dev/null 2>&1; then
      gdown "https://drive.google.com/uc?id=1HK9VE4r4SqsCEApaI9tW8V8vodzbS4uW" \
        -O "${AM_RAW_TAR}"
    else
      echo "[Ancient-Modern] 未找到 gdown，将尝试使用 curl 直接下载。"
      echo "                 如果失败，你可以手动从以下链接下载再放到 raw_datasets/:"
      echo "                 https://drive.google.com/file/d/1HK9VE4r4SqsCEApaI9tW8V8vodzbS4uW/view"
      echo
      curl -L \
        'https://drive.google.com/uc?export=download&id=1HK9VE4r4SqsCEApaI9tW8V8vodzbS4uW' \
        -o "${AM_RAW_TAR}" || {
          echo "[Ancient-Modern] curl 下载失败，请手动下载并放到: ${AM_RAW_TAR}"
          exit 1
        }
    fi

    echo "[Ancient-Modern] 下载完成: ${AM_RAW_TAR}"
  fi

  echo "[Ancient-Modern] 开始解压到 ${AM_OUT_DIR} ..."
  tar xf "${AM_RAW_TAR}" -C "${AM_OUT_DIR}"
  echo "[Ancient-Modern] 解压完成。请查看目录结构以确认包含 train/dev/test 等文件。"
fi

echo
echo "全部完成 ✅"
echo "StoryTrans 路径:    ${STORY_DIR}"
echo "Ancient-Modern 路径: ${AM_OUT_DIR}"
