"""从 reference/ 生成读者战场包 minivllm/（挖空版）。

做法：逐文件做"行级手术"——
- 模块 docstring、import、常量、类定义、数据类字段：原样保留；
- GIVEN 名单里的函数、dunder、@property：原样保留（这些是"脚手架"，直接给）；
- 其余函数：保留签名和 docstring，函数体换成 raise NotImplementedError("TODO")。

用法：
    .venv/bin/python scripts/make_stubs.py
"""

import ast
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
OUT = ROOT / "minivllm"

# 整文件原样照抄的模块（纯数据/脚手架，没有要读者填的逻辑）
GIVEN_FILES = {"scheduler/request.py"}

# 模块 -> 直接给出的函数/方法（限定名）。除此之外的函数都会被挖空。
GIVEN = {
    "tensor_ops.py": {"best_device"},
    "generate.py": {"count_params"},
    "model/attention.py": set(),
    "model/transformer.py": {
        "MultiHeadAttention._split_heads",
        "MultiHeadAttention._merge_heads",
        "MiniTransformer._embed",
        "MiniTransformer._head",
    },
    "model/qwen3.py": {
        "Qwen3Attention._project_qkv",
        "Qwen3ForCausalLM.from_pretrained",
    },
    "cache/kv_cache.py": {"NaiveKVCache.seq_len", "NaiveKVCache.reset"},
    "cache/block_pool.py": {"BlockPool.num_free_blocks", "BlockPool.can_fit"},
    "cache/block_table.py": {"BlockTable.needed_blocks", "BlockTable.free"},
    "cache/paged_cache.py": {"PagedKVCache.gather_padded"},
    "scheduler/scheduler.py": {
        "Scheduler.add_request",
        "Scheduler.has_unfinished",
        "Scheduler._blocks_needed",
        "SchedulerOutput.is_empty",
    },
    "scheduler/batching.py": set(),
    "sampling/sampler.py": {
        "Sampler._generator_for",
        "Sampler.sample_one",
        "Sampler.drop_request",
    },
    "engine/llm_engine.py": {
        "LLMEngine.add_request",
        "LLMEngine.has_unfinished",
        "LLMEngine._make_output",
    },
    "engine/llm.py": {"LLM._tokenize"},
    "bench/metrics.py": {"format_report"},
}

TODO_MESSAGE = "TODO: 看 learning-guide 对应章节，把这个函数填出来"


def is_given(qualname, node, given_names):
    if qualname in given_names:
        return True
    name = qualname.rsplit(".", 1)[-1]
    if name.startswith("__") and name.endswith("__"):
        return True  # dunder 直接给
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "property":
            return True
    return False


def collect_stubs(tree, given_names):
    """收集要挖空的函数：(body_start_lineno, end_lineno)（1 起始，含两端）。"""
    spans = []

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + child.name
                if not is_given(qual, child, given_names):
                    body = child.body
                    start_idx = 0
                    # 保留 docstring
                    if (
                        isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        start_idx = 1
                    if start_idx < len(body):
                        spans.append(
                            (body[start_idx].lineno, body[-1].end_lineno, qual)
                        )
                    # 函数体整个被挖掉时，留个哨兵
                    if start_idx >= len(body):
                        pass
                # 不递归进函数内部：内部嵌套函数随函数体一起被挖掉
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")

    visit(tree)
    return spans


def stub_source(source, rel_path):
    tree = ast.parse(source)
    given_names = GIVEN.get(rel_path, set())
    spans = collect_stubs(tree, given_names)
    lines = source.splitlines(keepends=True)
    for start, end, qual in sorted(spans, reverse=True):
        indent = " " * (len(lines[start - 1]) - len(lines[start - 1].lstrip()))
        lines[start - 1 : end] = [
            f'{indent}raise NotImplementedError("{TODO_MESSAGE}")  # {qual}\n'
        ]
    return "".join(lines)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for src in sorted(REF.rglob("*.py")):
        rel = src.relative_to(REF)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        source = src.read_text()
        if src.name == "__init__.py" or str(rel) in GIVEN_FILES:
            dst.write_text(source)
            continue
        dst.write_text(stub_source(source, str(rel)))
        print(f"stubbed {rel}")


if __name__ == "__main__":
    main()
