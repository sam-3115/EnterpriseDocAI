import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    print(f"Launching Streamlit Frontend: {frontend_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(frontend_path), "--server.port", "8501"])

