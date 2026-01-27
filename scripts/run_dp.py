"""
Dynamic Programming Solver for Irrigation Scheduling
====================================================

Runs DP solver on small-N irrigation environment for exact solution.

Usage:
    python scripts/run_dp.py
"""

import sys
sys.path.insert(0, '.')

from src.agents.dp_solver import *

if __name__ == "__main__":
    # This will run the main function from dp_solver
    print("Running DP solver...")
