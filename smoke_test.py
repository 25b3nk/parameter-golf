"""
CPU smoke test for parameter-golf pipeline.
Validates data loading, model forward/backward, optimizer step, and quantization/save.
Does NOT require CUDA. Run with: python smoke_test.py
"""
import os, sys, time, io, zlib
os.environ.setdefault("DATA_PATH",       "./data/datasets/fineweb10B_sp1024")
os.environ.setdefault("TOKENIZER_PATH",  "./data/tokenizers/fineweb_1024_bpe.model")
os.environ.setdefault("VOCAB_SIZE",      "1024")
os.environ.setdefault("NUM_LAYERS",      "4")
os.environ.setdefault("MODEL_DIM",       "256")
os.environ.setdefault("NUM_HEADS",       "4")
os.environ.setdefault("NUM_KV_HEADS",    "2")
os.environ.setdefault("TRAIN_SEQ_LEN",   "64")

import torch
import sentencepiece as spm
from train_gpt import (
    Hyperparameters, GPT, Muon, TokenStream, DistributedTokenLoader,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    quantize_state_dict_int4_hadamard, dequantize_state_dict_int4_hadamard,
    QUANTIZATION_SCHEME,
    build_sentencepiece_luts, load_validation_tokens, eval_val,
    restore_low_dim_params_to_fp32, zeropower_via_newtonschulz5,
)

DEVICE = torch.device("cpu")
STEPS  = 10

def sep(title): print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ── 1. Hyperparameters ─────────────────────────────────────────
sep("1. Hyperparameters")
args = Hyperparameters()
args.train_batch_tokens = 512
args.train_seq_len      = 64
args.val_batch_size     = 512
args.val_loss_every     = 0          # skip val during loop
args.warmup_steps       = 2
args.warmdown_iters     = 3
args.iterations         = STEPS
print(f"  device={DEVICE}  steps={STEPS}  seq_len={args.train_seq_len}  batch_tokens={args.train_batch_tokens}")

# ── 2. Data loading ────────────────────────────────────────────
sep("2. Data loading")
loader = DistributedTokenLoader(args.train_files, rank=0, world_size=1, device=DEVICE)
x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps=8)
print(f"  x.shape={x.shape}  y.shape={y.shape}  dtype={x.dtype}  OK")

# ── 3. Model instantiation ─────────────────────────────────────
sep("3. Model instantiation")
model = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
).to(DEVICE)
restore_low_dim_params_to_fp32(model)
nparam = sum(p.numel() for p in model.parameters())
print(f"  params={nparam:,}  dtype=float32  OK")

# ── 4. Forward pass ────────────────────────────────────────────
sep("4. Forward pass")
model.train()
loss = model(x.long(), y.long())
print(f"  loss={loss.item():.4f}  OK")

# ── 5. Optimizer setup ─────────────────────────────────────────
sep("5. Optimizer setup (Muon + Adam)")
matrix_params, scalar_params, embed_params = [], [], []
for name, p in model.named_parameters():
    if p.ndim >= 2 and "tok_emb" not in name:
        matrix_params.append(p)
    elif "tok_emb" in name:
        embed_params.append(p)
    else:
        scalar_params.append(p)

optimizers = [
    Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
         backend_steps=args.muon_backend_steps),
    torch.optim.Adam(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps),
    torch.optim.Adam(embed_params,  lr=args.embed_lr,  betas=(args.beta1, args.beta2), eps=args.adam_eps),
]
print(f"  matrix_params={len(matrix_params)}  scalar_params={len(scalar_params)}  embed_params={len(embed_params)}  OK")

# ── 6. Training loop ───────────────────────────────────────────
sep(f"6. Training loop ({STEPS} steps)")
t0 = time.time()
for step in range(STEPS):
    x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps=8)
    loss = model(x.long(), y.long())
    loss.backward()
    for opt in optimizers:
        opt.step()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    if step % 5 == 0:
        print(f"  step={step:3d}  loss={loss.item():.4f}")
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s  ({elapsed/STEPS*1000:.0f}ms/step)")

# ── 7. Validation (BPB) ────────────────────────────────────────
sep("7. Validation (BPB metric)")
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, DEVICE)

# Load a tiny slice of val tokens (just enough for one batch)
from train_gpt import load_data_shard
from pathlib import Path
import glob
val_files = sorted(glob.glob(args.val_files))
if val_files:
    val_tokens_full = load_data_shard(Path(val_files[0]))
    # Slice to a small chunk
    needed = (args.val_batch_size // args.train_seq_len) * args.train_seq_len + 1
    val_tokens = val_tokens_full[:needed].contiguous()
    print(f"  val_tokens shape={val_tokens.shape}  (trimmed for smoke test)")

    # Monkey-patch args.val_files to point only to first shard; run eval
    val_loss, val_bpb = eval_val(
        args, model, rank=0, world_size=1, device=DEVICE,
        grad_accum_steps=8,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"  val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}  OK")
else:
    print("  SKIP — no val files found")

# ── 8. Quantize & save ─────────────────────────────────────────
sep(f"8. Quantization + artifact size  [scheme={QUANTIZATION_SCHEME}]")
state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
_qfn = quantize_state_dict_int4_hadamard if QUANTIZATION_SCHEME == "int4_hadamard" else quantize_state_dict_int8
_dqfn = dequantize_state_dict_int4_hadamard if QUANTIZATION_SCHEME == "int4_hadamard" else dequantize_state_dict_int8
quant_obj, stats = _qfn(state_dict)
buf = io.BytesIO()
torch.save(quant_obj, buf)
raw_bytes = buf.getvalue()
compressed = zlib.compress(raw_bytes, level=9)
print(f"  params={stats['param_count']:,}")
print(f"  payload_bytes={stats['int8_payload_bytes']:,}")
print(f"  compressed artifact size={len(compressed):,} bytes  ({len(compressed)/1e6:.2f} MB)")
print(f"  16MB limit OK: {len(compressed) < 16_000_000}")

# ── 9. Dequantize & reload ─────────────────────────────────────
sep("9. Dequantize & reload")
buf.seek(0)
loaded_obj = torch.load(buf, weights_only=False)
recovered = _dqfn(loaded_obj)
model2 = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
).to(DEVICE)
model2.load_state_dict(recovered, strict=True)
model2.eval()
with torch.inference_mode():
    loss2 = model2(x.long(), y.long())
print(f"  dequantized model forward loss={loss2.item():.4f}  OK")

sep("ALL CHECKS PASSED")
print("  Pipeline is ready. Move to GPU when ready.")
