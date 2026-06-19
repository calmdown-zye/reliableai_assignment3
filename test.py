# test.py
"""
Runs Marabou robustness verification on a small MNIST FC model.
Verifies that all inputs within L-inf ball of radius epsilon=0.01
around a test image are classified as the same digit.

Usage: python3 test.py
"""
import subprocess
result = subprocess.run(
    ["python3", "scripts/verify.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)