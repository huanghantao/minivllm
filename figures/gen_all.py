"""统一运行全部生图脚本：python figures/gen_all.py"""

import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SKIP = {"gen_all.py", "_common.py"}


def main():
    scripts = sorted(p for p in HERE.glob("gen_*.py") if p.name not in SKIP)
    for script in scripts:
        print(f"[gen_all] {script.name}")
        runpy.run_path(str(script), run_name="__main__")
    print(f"[gen_all] 完成，共 {len(scripts)} 个脚本")


if __name__ == "__main__":
    main()
