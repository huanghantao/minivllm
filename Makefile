# minivllm 常用命令
# IMPL 环境变量决定测试打的是哪个实现：
#   默认打 minivllm（读者战场，未填时是红的）
#   IMPL=reference 打参考答案（应该全绿）

PY = .venv/bin/python
PYTEST = .venv/bin/pytest

.PHONY: test test-ref test-fast figures book serve

test:
	$(PYTEST) tests

test-ref:
	IMPL=reference $(PYTEST) tests

# 跳过需要真实权重的慢测试（日常开发用）
test-fast:
	IMPL=reference $(PYTEST) tests -m "not slow"

figures:
	$(PY) figures/gen_all.py

book:
	mdbook build

serve:
	mdbook serve --open
