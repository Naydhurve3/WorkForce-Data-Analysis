"""Launch the Streamlit dashboard."""

import subprocess
import sys


def main():
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
