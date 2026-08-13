import subprocess
import sys
import os

os.chdir(r"c:\Users\vaibh\OneDrive\Documents\Tracify\ml-engine")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
print(f"\nReturn code: {result.returncode}")
