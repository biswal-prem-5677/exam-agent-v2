"""
run.py – Unified launcher for the Autonomous Exam Proctoring & Grading Agent.
Starts both the FastAPI exam server (port 8000) and the Streamlit examiner console (port 8501).
Gracefully shuts down both processes on Ctrl+C.
"""
import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = Path(__file__).parent.resolve()
PYTHON = sys.executable

def main():
    print("=" * 65)
    print(">> Starting Autonomous Exam Proctoring & Grading Agent 2.0")
    print("=" * 65)
    
    # Environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE)
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # 1. Start FastAPI server via uvicorn
    cmd_api = [
        PYTHON, "-m", "uvicorn",
        "server.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "warning"
    ]
    print("[API] Launching FastAPI Server on http://localhost:8000 ...")
    proc_api = subprocess.Popen(cmd_api, cwd=str(BASE), env=env)

    # 2. Start Streamlit examiner console
    cmd_console = [
        PYTHON, "-m", "streamlit", "run",
        "console.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    print("[CONSOLE] Launching Streamlit Console on http://localhost:8501 ...")
    proc_console = subprocess.Popen(cmd_console, cwd=str(BASE), env=env)

    time.sleep(2)
    print("-" * 65)
    print("[READY] System Ready:")
    print("   * Student Exam Portal : http://localhost:8000/exam")
    print("   * Examiner Dashboard  : http://localhost:8501")
    print("   * Press Ctrl+C to stop both servers.")
    print("-" * 65)

    def shutdown(signum=None, frame=None):
        print("\n[STOP] Shutting down servers...")
        try:
            proc_api.terminate()
            proc_console.terminate()
            proc_api.wait(timeout=3)
            proc_console.wait(timeout=3)
        except Exception:
            proc_api.kill()
            proc_console.kill()
        print("[DONE] All services terminated cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
            # If either process died unexpectedly
            if proc_api.poll() is not None:
                print("[ERROR] FastAPI server stopped unexpectedly.")
                shutdown()
            if proc_console.poll() is not None:
                print("[ERROR] Streamlit console stopped unexpectedly.")
                shutdown()
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()

