# Parameter Golf — Autoresearch Program

This is an autonomous research loop adapted from Karpathy's `autoresearch` for the
OpenAI Parameter Golf challenge.

**Goal**: Achieve the lowest `val_bpb` on FineWeb, within a 16MB artifact (int8+zlib),
training in ≤10 min on 8xH100. Eval budget is an additional ≤10 min.

---

## Setup

Before starting a new experiment session:

1. **Agree on a run tag** (e.g. `exp-mar21`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read these files for full context**:
   - `README.md` — challenge rules, leaderboard, submission process.
   - `train_gpt.py` — the training script you will modify (≤1500 lines).
   - `autoresearch/program.md` — this file.
   - Latest SOTA record in `records/track_10min_16mb/` — understand what's already been tried.
4. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row.
5. **Confirm data exists**: `ls data/datasets/fineweb10B_sp1024/` should show train+val shards.
   If not, run: `python3 data/cached_challenge_fineweb.py --variant sp1024`
6. **Confirm setup** and start experimenting.

---

## Experiment Mechanics

### What you CAN modify
- `train_gpt.py` — model architecture, optimizer, hyperparameters, training loop, quantization.
  Everything is fair game: architecture, optimizer design, eval strategy, compression.
  Hard limit: **≤1500 lines**.

### What you CANNOT modify
- `data/` — dataset and tokenizer are fixed. BPB must be computed over the same val split.
- Competition rules: no external downloads during eval, artifact ≤16MB (code bytes + compressed model bytes).

### Running an experiment (cheap iteration on 1xH100)

```bash
cd /workspace/parameter-golf
RUN_ID=exp_$(date +%s) \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py > autoresearch/run.log 2>&1
```

Using `MAX_WALLCLOCK_SECONDS=300` (5 min) on 1xH100 for cheap iteration.
When a technique is confirmed to work, re-run with `nproc_per_node=8 MAX_WALLCLOCK_SECONDS=600`
to get the true 10-minute 8xH100 leaderboard score.

### Extracting results

```bash
grep "final_int8_zlib_roundtrip val" autoresearch/run.log | tail -1
grep "val_bpb" autoresearch/run.log | tail -5
```

---

## Logging results

After each experiment, log to `autoresearch/results.tsv` (tab-separated, NOT comma-separated).

Header row:
```
commit	val_bpb	memory_mb	status	gpus	description
```

- `commit`: 7-char git hash
- `val_bpb`: final `final_int8_zlib_roundtrip` val_bpb (0.000000 for crashes)
- `memory_mb`: peak CUDA memory in MiB (0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `gpus`: number of GPUs used (1 for cheap tests, 8 for leaderboard runs)
- `description`: short description of what changed

---

## Research Agenda

Work through this list roughly in priority order. These are the most promising
unexplored directions based on analysis of autoresearch (Karpathy, 2026) and the
current leaderboard.

### Tier 1 — Highest expected impact (not yet in baseline)

1. **ResFormer Value Embeddings**
   Alternating-layer `nn.Embedding(vocab_size, kv_dim)` mixed into attention values
   via an input-dependent sigmoid gate. Gate initialized to zero (neutral).
   At vocab=1024, the embeddings cost only ~1024 * kv_dim * n_ve_layers bytes.
   Implementation:
   ```python
   # In GPT.__init__, for alternating layers:
   self.value_embeds = nn.ModuleDict({
       str(i): nn.Embedding(args.vocab_size, kv_dim)
       for i in range(args.num_layers) if i % 2 == (args.num_layers-1) % 2
   })
   # Gate per attention layer (small):
   self.ve_gate = nn.Linear(32, num_kv_heads, bias=False)  # initialized to zeros
   # In attention forward:
   if str(layer_idx) in model.value_embeds:
       ve = value_embeds[str(layer_idx)](input_ids).view(B, T, n_kv_heads, head_dim)
       gate = 2 * torch.sigmoid(ve_gate(x[..., :32]))
       v = v + gate.unsqueeze(-1) * ve
   ```

2. **NorMuon (variance-normalized Muon)**
   After orthogonalization, normalize per-direction variance using a second momentum buffer.
   Keeps update magnitude consistent across layers with different gradient scales.
   Reduces instability with small models.

3. **Polar Express Orthogonalization**
   Replace the Newton-Schulz5 iteration with Polar Express coefficients (4-step):
   ```python
   polar_express_coeffs = [
       (8.156554524902461, -22.48329292557795, 15.878769915207462),
       (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
       (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
       (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
   ]
   ```
   Batch same-shape matrices together: `stacked = torch.stack([p for p in same_shape_params])`
   then orthogonalize in one kernel call.

4. **Decaying Weight Decay**
   Linearly decay Muon WD from `WEIGHT_DECAY` to 0 by end of training:
   ```python
   progress = elapsed_ms / max_wallclock_ms
   muon_weight_decay = args.muon_weight_decay * (1 - progress)
   for group in optimizer_muon.param_groups:
       group["weight_decay"] = muon_weight_decay
   ```

5. **Cautious Weight Decay**
   Apply WD only where gradient and param agree in sign:
   ```python
   mask = (g * stacked_params) >= 0
   stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
   ```

### Tier 2 — Medium impact

6. **x0 Residual Mixing (per-layer scalars)**
   Learned mix of current residual stream `x` with the original embedding `x0`.
   Already partially present as `resid_mix` in the baseline — verify it's optimal.
   Karpathy's version: separate `resid_lambdas` (init 1.0) and `x0_lambdas` (init 0.1).

7. **GC disable after step 0**
   ```python
   import gc
   # After first step:
   if step == 1:
       gc.collect(); gc.freeze(); gc.disable()
   ```
   Eliminates ~500ms GC stalls during training = more steps within 10 min.

8. **LR Scaling by model dim**
   Scale all LRs by `(model_dim / 768) ** -0.5` to be robust to architecture changes.

9. **Stacked Muon (batch same-shape matrices)**
   Group all matrix params of the same shape and orthogonalize together with
   `torch.compile(fullgraph=True)`. Reduces optimizer overhead on 8xH100.

10. **Logit softcap tuning**
    Current default: 30. Karpathy uses 15. Try values between 10-30.

### Tier 3 — Architecture exploration

11. **QK normalization after RoPE** (already in baseline as `q_gain`)
    Verify the current `q_gain` param is equivalent or try explicit `F.rms_norm` on q,k.

12. **Sliding window pattern "SSSL"**
    Already used in SOTA. Confirm still optimal at current depth.

13. **Deeper + narrower vs. shallower + wider**
    Try 12L×384dim vs 10L×512dim vs 8L×640dim under same 16MB constraint.

14. **MLP activation: relu^2 vs swiglu vs geglu**
    Current: relu^2. SwiGLU may help but costs more params per layer.

15. **Bias terms on final norm / head**
    Cheap params that sometimes help small models.

---

## The Experiment Loop

LOOP FOREVER:

1. Check `git log --oneline -5` and `autoresearch/results.tsv` to understand current state.
2. Pick the next idea from the research agenda above (or your own idea).
3. Modify `train_gpt.py` directly — one change at a time.
4. `git commit -m "exp: <description>"`
5. Run: `... torchrun ... train_gpt.py > autoresearch/run.log 2>&1`
6. Extract val_bpb: `grep "final_int8_zlib_roundtrip val" autoresearch/run.log`
7. If empty: `tail -50 autoresearch/run.log` — fix crash or log as crash.
8. Check artifact size: `grep "Total submission size int8" autoresearch/run.log`
   — must be <16,000,000 bytes.
9. Log to `autoresearch/results.tsv`.
10. If improved: advance the branch (keep commit).
    If not improved: `git reset --hard HEAD~1` (revert).

**Timeout**: If a run exceeds 8 minutes total, kill it and treat as crash.

**NEVER STOP**: Run indefinitely until manually interrupted. If out of ideas, re-read records/
for inspiration, combine near-misses, try more radical changes. The human may be asleep.

**Crashes**: If crash is from a fixable bug (typo, missing import), fix and re-run.
If the idea is fundamentally broken, log crash and move on.

**Simplicity criterion**: A 0.001 val_bpb gain that adds 30 ugly lines? Probably not worth it.
A 0.001 gain from a clean 5-line change? Keep. An improvement of ~0 but simpler code? Keep.

---

## Current SOTA

As of 2026-03-19, best leaderboard entry: **1.1748 BPB**
Key techniques in SOTA: sliding window eval + FP16 embed + 10L + Muon WD + Overtone init + resid mix

Beat it by at least 0.005 nats to qualify for a new record submission.
Need ≥3 seed runs to demonstrate statistical significance (p < 0.01).
