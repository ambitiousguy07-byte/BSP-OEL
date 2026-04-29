import subprocess
import sys
import os

def launch():
    print("🚀 Initializing Bio-Signal Pro Suite...")
    try:
        # Launch streamlit using the current python executable
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n✅ Dashboard shutdown successfully.")

if __name__ == "__main__":
    launch()
