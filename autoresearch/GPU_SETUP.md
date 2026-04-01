# GPU Setup Guide

## Quick start on RunPod (recommended)

1. Deploy a **1xH100 SXM pod** using the official Parameter Golf template:
   https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
   (All Python deps are pre-installed in the image)

2. SSH in and clone:
   ```bash
   cd /workspace
   git clone https://github.com/openai/parameter-golf.git
   cd parameter-golf
   ```

   Or if using this fork, push first and clone your branch.

3. Download full dataset (80 shards, ~8B tokens):
   ```bash
   python3 data/cached_challenge_fineweb.py --variant sp1024
   ```
   For quick iteration on 1xH100, 1-5 shards is enough:
   ```bash
   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 5
   ```

4. Run baseline to verify setup:
   ```bash
   ./autoresearch/run.sh --gpus 1 --time 300
   # Expected: val_bpb ~1.26 (1xH100, 5 min, baseline config)
   ```

---

## Iteration workflow

### Cheap experiments (1xH100, ~$3/hr)
```bash
./autoresearch/run.sh --gpus 1 --time 300
```
Use for testing new ideas. Results are directionally valid but BPB will be worse
than the full 8xH100 run (fewer steps in the same 5-min budget).

### Final leaderboard run (8xH100, ~$20/hr — use sparingly)
```bash
./autoresearch/run.sh --gpus 8 --time 600
```
Only run this once a technique is confirmed to help on 1xH100.
Run 3× with different seeds to prove statistical significance.

---

## Autoresearch autonomous loop

On your GPU machine, start a Claude Code session in this repo and prompt:
```
Hi, have a look at autoresearch/program.md and kick off a new experiment session.
Let's do the setup first.
```

Let it run overnight. Each 1xH100 experiment takes ~5-7 min (including compile).
~10-12 experiments per hour → ~80-100 experiments per 8-hour sleep.

---

## Submitting a new record

Once you have a val_bpb that beats SOTA by ≥0.005 nats:

1. Create a folder under `records/track_10min_16mb/`:
   ```
   records/track_10min_16mb/YYYY-MM-DD_<ShortDescription>/
   ├── README.md          # Explain what you changed and why it works
   ├── train_gpt.py       # Self-contained snapshot of the training script
   ├── train.log          # Full training log (all 3 seed runs)
   └── submission.json    # See existing records for format
   ```

2. Run 3 seeds to prove reproducibility:
   ```bash
   for seed in 1337 42 123; do
     SEED=$seed ./autoresearch/run.sh --gpus 8 --time 600 --id "record_seed${seed}"
     cp autoresearch/run.log records/track_10min_16mb/<folder>/train_seed${seed}.log
   done
   ```

3. Open a PR against `openai/parameter-golf` on GitHub.

---

## Key metrics to track

From `autoresearch/run.log`:
```bash
# Final BPB (what gets submitted):
grep "final_int8_zlib_roundtrip val" autoresearch/run.log

# Artifact size (must be <16,000,000 bytes):
grep "Total submission size int8" autoresearch/run.log

# Training time (must be <600s):
grep "stopping_early\|train_time" autoresearch/run.log | tail -3

# Peak memory:
grep "peak memory" autoresearch/run.log
```
