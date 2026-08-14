#!/bin/bash
# 组装 mdbook 的 staging 目录 .book-src/（mdbook 会拷贝 src 下的一切，
# 所以把书需要的内容挑出来，避开 .venv/.git 等大目录）。
# 用法：scripts/stage_book.sh && mdbook build   或   && mdbook serve --open
set -e
cd "$(dirname "$0")/.."
rm -rf .book-src
mkdir -p .book-src/figures
cp SUMMARY.md README.md .book-src/
cp -r learning-guide .book-src/
cp -r figures/out .book-src/figures/
echo "staged .book-src/"
