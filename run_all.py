"""
run_all.py — Run the complete pipeline in one go
"""
import subprocess, sys

print("=" * 60)
print("  RUNNING FULL ML COMPARISON PIPELINE")
print("=" * 60)

steps = [
    ("Data Exploration & Preprocessing", "data/preprocessing.py"),
    ("Main Model Training & Evaluation",  "main.py"),
    ("Hyperparameter Tuning",             "models/train_evaluate.py"),
]

for label, script in steps:
    print(f"\n▶  {label} …")
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        sys.exit(1)

print("\n" + "=" * 60)
print("  ALL STEPS COMPLETE")
print("  Results saved in:  results/")
print("  Launch UI with:    python ui/app.py")
print("=" * 60)
