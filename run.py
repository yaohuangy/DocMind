"""
Docmind - 启动入口。

一键启动 Streamlit 应用。
"""

import subprocess
import sys
from pathlib import Path


def main():
    """启动 Streamlit 应用。"""
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"错误: 找不到 app.py ({app_path})")
        sys.exit(1)

    print("=" * 60)
    print("  📚 Docmind")
    print("=" * 60)
    print()
    print(f"  启动地址: http://localhost:7860")
    print(f"  应用路径: {app_path}")
    print()
    print("  按 Ctrl+C 停止服务")
    print("=" * 60)

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.address", "0.0.0.0",
            "--server.port", "7860",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
