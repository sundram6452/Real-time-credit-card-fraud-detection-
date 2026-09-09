"""
Main Application Entrypoint
Provides unified execution for the Fraud Detection Platform.
Running `python main.py` directly launches the Streamlit Analytics Dashboard by default.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import main as run_cli

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default behavior when run with no arguments: start dashboard
        print("==================================================================")
        print("  FRAUDGUARD AI - REAL-TIME CREDIT CARD FRAUD DETECTION PLATFORM  ")
        print("==================================================================")
        print("No command arguments provided. Launching Streamlit Dashboard on port 8501...")
        print("To see all CLI options, run: python main.py --help")
        print("==================================================================\n")
        sys.argv.append("dashboard")
    
    run_cli()
