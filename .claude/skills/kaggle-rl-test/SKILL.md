---
name: kaggle-rl-test
description: |
  End-to-end workflow for testing and training PPO reinforcement learning for robot navigation on Kaggle's free GPU,
  using the Kaggle API from the command line (compatible with Claude Code control).

  Trigger this skill whenever the user wants to:
  - train PPO policy for robot navigation on Kaggle free GPU
  - run longer training experiments that need GPU compute
  - benchmark curriculum learning vs vanilla PPO
  - push training scripts to Kaggle for GPU acceleration
  - retrieve training logs and trained checkpoints after run
  - iterate on RL policy design using Kaggle as free compute backend

  Also trigger for phrases like: "免费GPU训练PPO", "kaggle跑训练", "train on kaggle",
  "用kaggle训练策略", "kaggle免费训练", "提交任务到kaggle".

  Always use this skill — don't try to reconstruct the workflow from scratch.
---

# Kaggle GPU — PPO RL Navigation Training Skill

Trains **PPO for Occupancy Grid Robot Navigation** on Kaggle's free T4 GPU.
Full loop: local code → push → train → pull logs/checkpoints, all via CLI (Claude Code friendly).

---

## Prerequisites (Already Done)

- ✅ Kaggle CLI already installed
- ✅ API authentication already configured

---

## Recommended Approach: Source Code via Kaggle Dataset

This approach avoids GitHub DNS resolution issues by packaging source as a Kaggle Dataset.

### Step 1 — Create source zip and upload as Dataset

From your project root:

```bash
cd /home/xpeng/Documents/robot/xrollout/projects/rl

# Create source zip (exclude git, cache, large files)
zip -r occ_rl_src.zip . -x "*.git*" "__pycache__/*" "*.pyc" "kaggle-working/*" "checkpoints/*" "*.log" "results/*" "output/*"

# Create dataset directory
mkdir -p kaggle-dataset && cd kaggle-dataset
cp ../occ_rl_src.zip .

# Create dataset metadata
cat > dataset-metadata.json << 'EOF'
{
  "title": "occ-rl-source",
  "id": "YOUR_KAGGLE_USERNAME/occ-rl-source",
  "licenses": [{"name": "other"}]
}
EOF

# Create and upload the dataset
kaggle datasets create -p .
```

> **Replace `YOUR_KAGGLE_USERNAME`** with your actual Kaggle username.

---

### Step 2 — Create kernel with dataset reference

```bash
cd /home/xpeng/Documents/robot/xrollout/projects/rl
mkdir -p kaggle-working && cd kaggle-working
kaggle kernels init -p .
```

Create `kernel-metadata.json` with dataset reference:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/ppo-navigation-curriculum",
  "title": "ppo-navigation-curriculum",
  "code_file": "run_train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": ["YOUR_KAGGLE_USERNAME/occ-rl-source"],
  "competition_sources": [],
  "kernel_sources": []
}
```

---

### Step 3 — Create requirements.txt

```txt
# Core dependencies
numpy>=1.20.0
scipy>=1.7.0

# RL training
torch>=1.10.0
gymnasium>=0.26.0

# Environment
pettingzoo>=1.22.0

# Visualization
matplotlib>=3.4.0
opencv-python>=4.5.0
pygame>=2.1.0
imageio>=2.9.0

# Utils
tensorboard>=2.8.0
tqdm>=4.62.0
pyyaml>=5.4.0
```

---

### Step 4 — Create run_train.py

This unzips the source from Kaggle Dataset and starts training:

```python
# Kaggle automatically installs from requirements.txt
import zipfile
import os
import sys

# Unzip source code from the uploaded dataset
ZIP_PATH = "/kaggle/input/occ-rl-source/occ_rl_src.zip"
EXTRACT_DIR = "/kaggle/working/occ_rl"

print(f"Extracting source from {ZIP_PATH} to {EXTRACT_DIR}...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

# Change to project directory
os.chdir(EXTRACT_DIR)

# Add project to Python path
sys.path.insert(0, EXTRACT_DIR)

# Print GPU info
import torch
print(f"\n=== GPU Information ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Fix hardcoded user path in train_curriculum_final.py
fixed_path_file = "training/train_curriculum_final.py"
with open(fixed_path_file, "r") as f:
    content = f.read()
content = content.replace(
    "sys.path.insert(0, '/Users/bobinding/Documents/robot/xrollout')",
    f"sys.path.insert(0, '{EXTRACT_DIR}')"
)
with open(fixed_path_file, "w") as f:
    f.write(content)

print()

# === AVAILABLE TRAINING COMMANDS ===
# Option 1: Vanilla PPO custom implementation (hard scenario)
# from training.train_ppo_custom import main
# main()

# Option 2: Final 4-Phase Curriculum Learning (recommended)
# 4 phases: 0 → 2 → 4 → 5+2 obstacles, total 700K steps
from training.train_curriculum_final import main

print(f"Running: train_curriculum_final.py (4-phase curriculum)")
print("-" * 60)

main()

print("\n=== Training Complete ===")
print(f"Logs/checkpoints saved to: ./curriculum_v2_output/")
```

---

### Step 5 — Push and run on Kaggle

```bash
cd kaggle-working
kaggle kernels push -p .
```

---

### Step 6 — Monitor status

```bash
# Check status
kaggle kernels status YOUR_KAGGLE_USERNAME/ppo-navigation-curriculum
```

Or watch:

```bash
watch -n 30 kaggle kernels status YOUR_KAGGLE_USERNAME/ppo-navigation-curriculum
```

**Status progression**: `queued` → `running` → `complete` / `error`

Stream live logs:

```bash
mkdir -p ./output
kaggle kernels output YOUR_KAGGLE_USERNAME/ppo-navigation-curriculum -p ./output
cat ./output/*.log 2>/dev/null || echo "No log output yet, wait a bit..."
```

---

### Step 7 — Pull results back locally

```bash
# Create output directory
mkdir -p ./results

# Pull all output
kaggle kernels output YOUR_KAGGLE_USERNAME/ppo-navigation-curriculum -p ./results
```

This downloads:
- **Phase checkpoints + final model** → `./results/occ_rl/curriculum_v2_output/`
- **Script output logs** → `./results/*.log`

Copy final model to local:

```bash
cp ./results/occ_rl/curriculum_v2_output/final_model.pt /home/xpeng/Documents/robot/xrollout/projects/rl/checkpoints/curriculum_final_kaggle.pt
```

---

### Step 8 — Evaluate locally

```bash
cd /home/xpeng/Documents/robot/xrollout/projects/rl
python scipts/evaluate_policy.py --checkpoint checkpoints/curriculum_final_kaggle.pt --episodes 50
```

---

## Alternative: Direct Push (Small Codebase)

If you don't want to use the Dataset approach, you can push source directly with the kernel:

1. Copy all `*.py` files locally into `kaggle-working/`:
   ```bash
   cp -r envs policies training kaggle-working/
   ```

2. Use the simpler `run_train.py` that doesn't need to clone:
   ```python
   import os
   import sys
   import torch

   # Code already included
   sys.path.insert(0, os.path.abspath("."))

   # Print GPU info... (see above)

   from training.train_curriculum_final import main
   main()
   ```

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `401 Unauthorized` on push | Verify API token: `echo $KAGGLE_API_TOKEN` or check `~/.config/kaggle/kaggle.json` permissions (needs `chmod 600`) |
| `CUDA out of memory` | The final curriculum uses batch size 64 which should fit in T4. If OOM still happens, reduce batch size in the code |
| `Kernel status: error` with no log | Check `kernel-metadata.json` — `enable_gpu` must be `true` |
| `ModuleNotFoundError: torch` | Normal — Kaggle installs from requirements.txt at startup, just wait for the install to complete |
| `Git clone fails / Could not resolve host github.com` | Use the **Dataset approach** above, it doesn't need internet/GitHub |
| `FileNotFoundError: training/...py` | Verify all source files were included in the zip/dataset |
| Training exceeds Kaggle 9-hour time limit | The final curriculum is ~700K steps which should complete in ~4-6 hours on T4 |

---

## Quick Iteration Workflow

1. After changing code locally, re-create the zip and update the dataset:
   ```bash
   cd /path/to/project
   zip -r occ_rl_src.zip . -x "*.git*" "__pycache__/*" "*.pyc" "kaggle-working/*" "checkpoints/*"
   cp occ_rl_src.zip kaggle-dataset/
   cd kaggle-dataset
   kaggle datasets version -m "update source" -p .
   ```
2. Then re-push the kernel:
   ```bash
   cd ../kaggle-working
   kaggle kernels push -p .
   ```
3. Keep everything private so you don't spam public listings.

---

## Expected Output for 4-Phase Curriculum

A successful run on T4 GPU will look like:

```
================================================================================
CURRICULUM LEARNING v2
================================================================================

================================================================================
Phase 1: Easy (0 obstacles)
Target timesteps: 100,000
================================================================================
Update  10: Reward=  45.32, Success=35/50
...

Evaluating Phase 1: Easy (0 obstacles)...
Success Rate: 82.0% (41/50)
Saved: curriculum_v2_output/phase_1.pt

... (phases 2-4 output) ...

================================================================================
CURRICULUM LEARNING COMPLETE!
Total timesteps: 700,000
Best success rate: 74.0%
Final model: curriculum_v2_output/final_model.pt
================================================================================
```

**Expected total time**: ~4-5 hours on T4 GPU, well within Kaggle's 9-hour limit.

---

## Cleanup Local Files

```bash
# Clean up output to save space
rm -rf kaggle-working/output/*
rm -rf kaggle-working/results/*
rm occ_rl_src.zip  # if you don't need it anymore
```
