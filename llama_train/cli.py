"""train.py's command line: the flags, and the defaults that depend on other flags."""

import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hub repo the saved model is meant for (pushed separately with push_model.py)")
    parser.add_argument("--token_budget", type=float, default=6e9, help="Training tokens (lab cap 6B; every token of a step counts, there is no padding). Sets the step count")
    parser.add_argument("--learning_rate", type=float, default=None, help="Peak LR; with muon / muonium the AdamW (embedding, norms) LR. Default: 3e-3 for muonium (best run), 6e-4 otherwise (untuned starting point for 1B at ~0.5M-token batches)")
    parser.add_argument("--warmup_frac", type=float, default=0.02, help="Linear warmup over this fraction of the steps")
    parser.add_argument("--decay_frac", type=float, default=0.1, help="--lr_schedule wsd: 0 = constant after warmup; >0 = linear decay to 0 over this fraction of the final steps (default 0.1, MiniCPM WSD; ties 0.2 for Sophia at 1.1B, experiments.md 2026-10-14; runs before that used 0.2)")
    parser.add_argument("--decay_type", type=str, default="linear", choices=["linear", "1-sqrt", "cosine"], help="--lr_schedule wsd (step units): shape of the final decay to 0 (transformers get_wsd_schedule; 1-sqrt: lr x (1 - sqrt(progress)), Hagele et al. 2024)")
    parser.add_argument("--lr_schedule", type=str, default="wsd", choices=["wsd", "wsd2", "cosine"], help="wsd (warmup-stable-decay, see --decay_frac), wsd2 (warmup, stable, linear decay to --stable2_lr_ratio over --decay1_frac, stable2 at that LR for --stable2_frac, then linear decay to 0 over --decay_frac) or cosine (Llama 2/3: warmup, then cosine to --min_lr_ratio x peak)")
    parser.add_argument("--decay1_frac", type=float, default=0.05, help="--lr_schedule wsd2: first decay (peak -> --stable2_lr_ratio x peak) over this fraction of the steps")
    parser.add_argument("--stable2_frac", type=float, default=0.1, help="--lr_schedule wsd2: second stable stage at --stable2_lr_ratio x peak, fraction of the steps")
    parser.add_argument("--stable2_lr_ratio", type=float, default=0.1, help="--lr_schedule wsd2: LR of the second stable stage / peak")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="--lr_schedule cosine: final LR / peak (Llama 2: 0.1)")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon (Llama 2: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (Llama 3 / GPT-3: 0.1)")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2 for both optimizers (0.95: Llama/GPT-3; 0.999: PyTorch default)")
    parser.add_argument("--block_size", type=int, default=8192, help="Context window / training block (8192 = DESCRIPTION.md; train_scorer.py: 2048)")
    parser.add_argument("--config_overrides", type=str, default=None, help='JSON dict applied to the Llama 3.2 1B config with config.update, e.g. a smaller model (train_scorer.py)')
    parser.add_argument("--global_batch_size", type=int, default=None, help="Sequences of 8192 tokens per optimizer step (per-GPU batch x grad accum x GPUs); 64 = 0.5M tokens. Default 64; giving it turns off the default --batch_ramp")
    parser.add_argument("--batch_ramp", type=str, default=None, help='Default "32:0.5,64" (switch-point sweeps at 1B and 3B) unless --global_batch_size is given; "none" for a fixed batch. Global batch-size ramp "B1:t1,B2:t2,...,Bn": batch Bi for the next ti tokens, the last for the rest. ti > 1 is a token count ("32:1e9,64", or with a K/M/B suffix: "32:1B,64"), ti <= 1 a fraction of the token budget ("32:0.1,64"). Grad accum is fixed (enough for the largest Bi at --per_device_batch), so the micro-batch is Bi/(GPUs x accum); overrides --global_batch_size')
    parser.add_argument("--ramp_lr_mult", type=float, default=1.0, help="--batch_ramp, --optimizer muonium: at each batch switch multiply the LR of the Muon / NorMuon groups by this (cumulative over switches; e.g. 1.414 = sqrt of a 2x batch step). The WSD shape continues from the new peak; AdamW groups keep their LR")
    parser.add_argument("--schedule_unit", type=str, default="steps", choices=["steps", "tokens"], help="--lr_schedule wsd: warmup / decay lengths in steps (fractions of the step count) or in tokens (fractions of --token_budget; the LR at a given token count then does not depend on the batch ramp, so runs that share a prefix can branch from a checkpoint)")
    parser.add_argument("--save_at_tokens", type=str, default=None, help='Save a full checkpoint (model, optimizer, scheduler, RNG) at the last step at or before each of these token counts, e.g. "100M,200M,333M": the steps where --batch_ramp=32:<count>,64 would switch. Into --ckpt_dir/checkpoint-<step>, linked as tokens-<count>')
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint dir (default /work/$USER/llama_ckpt/<run_name>)")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to continue from (a --save_at_tokens checkpoint): restores model, optimizer, scheduler, RNG and the data position. The run's own flags (e.g. a different --batch_ramp after the branch point) apply from there")
    parser.add_argument("--per_device_batch", type=int, default=4, help="Max sequences per GPU per micro-batch; grad accum covers the rest")
    parser.add_argument("--fused_ce", action=argparse.BooleanOptionalAction, default=False, help="Liger fused LM head + cross-entropy: never materializes the full fp32 logits (128k vocab x 8192 tokens = 4 GiB per sequence). On in run.sbatch/profile.sbatch; --no-fused_ce disables")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_3", help="HF attention backend (default flash_attention_3: needed by --packing doc): sdpa (cuDNN flash), flash_attention_3 (Hub kernel kernels-community/vllm-flash-attn3 via `kernels`), flash_attention_4 (pip flash-attn-4, CuTe DSL)")
    parser.add_argument("--packing", type=str, default="doc", choices=["concat", "doc"], help="Default doc (~10%% faster, experiments.md 2026-10-14). concat: documents concatenated into 8192 blocks, attention runs across document boundaries. doc: same blocks, but each document attends only to itself and RoPE positions restart at each <|begin_of_text|> (FlashAttention varlen; needs --attn_implementation flash_attention_3). Applies to the evals too")
    parser.add_argument("--max_steps", type=int, default=None, help="Override the token-budget step count (e.g. short speed tests)")
    parser.add_argument("--num_evals", type=int, default=20, help="Evals over the run (lab: at least every 10%% of the steps)")
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=False, help="torch.compile each decoder layer (not the rotary/mask setup or the Liger loss): fuses RMSNorm/cast/SwiGLU/elementwise kernels. On in run.sbatch/profile.sbatch; --no-torch_compile disables")
    parser.add_argument("--profile_dir", type=str, default=None, help="Profile 5 steps with torch.profiler into this dir; works inside a full run (see --profile_start) or in profile.sbatch")
    parser.add_argument("--profile_start", type=int, default=8, help="Profile steps profile_start+1 .. profile_start+5 (pick a window without an eval step in a full run, e.g. 300)")
    parser.add_argument("--profile_cpu", action=argparse.BooleanOptionalAction, default=True, help="Also record CPU ops (shows what causes GPU gaps, slightly slows launches); --no-profile_cpu = GPU kernels only")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True, help="Periodic eval (--no-eval for short speed/profile jobs, see profile.sbatch)")
    parser.add_argument("--ddp_find_unused", action=argparse.BooleanOptionalAction, default=False, help="DDP find_unused_parameters. Trainer would set True (no gradient checkpointing), but Llama has no unused params; off is ~1%% faster (experiments.md)")
    parser.add_argument("--ddp_bf16_grads", action=argparse.BooleanOptionalAction, default=False, help="All-reduce gradients in bf16 (DDP bf16_compress_hook): half the NCCL traffic, grads stay fp32 locally. No gain with 200 MB buckets")
    parser.add_argument("--ddp_bucket_cap_mb", type=int, default=200, help="DDP gradient bucket size; 200 MB vs torch's 25 MB: ~5%% faster on 8 GPUs and ~10 GiB less memory (experiments.md)")
    parser.add_argument("--optimizer", type=str, default="muonium", choices=["adamw", "adam_mini", "muon", "muonium", "sophia", "soap"], help="adamw (torch AdamW), adam_mini (Adam-mini, pip adam-mini), muon (Muon for decoder-layer matrices + AdamW for the rest, muon_adamw.py), muonium (pip muonium: Muon for the decoder-layer matrices + its fused AdamW for embedding / norms, one LR = --learning_rate, wd = --weight_decay, package defaults for the rest), sophia (SophiaG, official sophia.py) or soap (official soap.py, work split across GPUs by soap_dist.py); lr = --learning_rate for sophia and soap")
    parser.add_argument("--soap_betas", type=str, default="0.95,0.95", help="SOAP beta1,beta2 (official defaults; beta2 also drives the preconditioner EMA). Epsilon: --adam_eps")
    parser.add_argument("--soap_precond_freq", type=int, default=10, help="SOAP: re-estimate the preconditioner eigenbases every k steps (official default 10)")
    parser.add_argument("--soap_impl", type=str, default="official", choices=["official", "meta"], help="official: soap.py (paper authors) split across GPUs by soap_dist.py. meta: facebookresearch distributed_shampoo with DefaultSOAPConfig (ZeRO-1 over DDP ranks, one AllGather of the updates, blocked preconditioners)")
    parser.add_argument("--soap_block", type=int, default=8192, help="--soap_impl meta: max_preconditioner_dim; larger dims are split into blocks (8192: no layer matrix is split, the embedding is; README suggests 1024-8192)")
    parser.add_argument("--soap_comm_dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="--soap_impl meta: dtype of the AllGather of the updates")
    parser.add_argument("--soap_max_precond_dim", type=int, default=10000, help="SOAP: dims above this are not preconditioned (official 10000: covers 8192, skips the 128k vocab)")
    parser.add_argument("--soap_precondition_1d", action=argparse.BooleanOptionalAction, default=False, help="--soap_impl official: also precondition the 1-D params (RMSNorm weights) with a full d x d matrix. Off (official default, paper sec. 4.1, chosen for overhead): plain Adam for them")
    parser.add_argument("--sophia_rho", type=float, default=0.05, help="SophiaG rho (clip threshold; official GPT-2 125M/770M: 0.05). Tune so train/sophia_win_rate stays in 0.1-0.5")
    parser.add_argument("--sophia_betas", type=str, default="0.965,0.99", help="SophiaG beta1,beta2 (official defaults)")
    parser.add_argument("--sophia_hess_interval", type=int, default=10, help="Re-estimate the diagonal Hessian (Gauss-Newton-Bartlett) every k steps (official: 10); one extra forward + backward of the full batch, i.e. ~1/k extra compute (official: same)")
    parser.add_argument("--muon_lr", type=float, default=None, help="--optimizer muon / muonium: peak Muon LR, default 0.01 for muonium (best run), 1.25e-3 for muon (muon: match_rms_adamw scaling, tuned on GPT-2; muonium: x sqrt(max(1, rows/cols)), Keller Jordan's scale ~0.02); --learning_rate is the AdamW / Sophia part's")
    parser.add_argument("--muonium_algorithm", type=str, default="normuon", choices=["muon", "normuon"], help="--optimizer muonium: update rule for the matrices (muon, or normuon: per-neuron second-moment normalization, muon_beta2 0.95)")
    parser.add_argument("--muonium_rest", type=str, default="adamw", choices=["adamw", "sophia"], help="--optimizer muonium: optimizer for the non-matrix params (tied embedding, norms) at --learning_rate / --weight_decay: muonium's AdamW, or SophiaG (--sophia_* flags, Hessian pass as in --optimizer sophia)")
    parser.add_argument("--muon_split_heads", type=str, default="none", choices=["none", "kv", "qkv"], help="--optimizer muonium: k_proj / v_proj (qkv: and q_proj) in their own Muon group, orthogonalized per attention head (muonium split_sizes, head_dim row blocks) instead of as one matrix")
    parser.add_argument("--muon_lr_mult_kv", type=float, default=1.0, help="--optimizer muonium: k_proj / v_proj in their own Muon group at --muon_lr x this")
    parser.add_argument("--norm_lr", type=float, default=None, help="--optimizer muonium (adamw rest): LR of the RMSNorm weights' AdamW group (default: --learning_rate, shared with the embedding)")
    parser.add_argument("--embed_optimizer", type=str, default="adamw", choices=["adamw", "soap"], help="--optimizer muonium (adamw rest): optimizer of the tied embedding. soap: official SOAP (--soap_* flags, LR --learning_rate, wd --weight_decay; the 128k vocab dim is not preconditioned) chained with muonium; give the norms their LR with --norm_lr")
    parser.add_argument("--muon_wd", type=float, default=None, help="--optimizer muonium: weight decay of the Muon matrices (default: --weight_decay)")
    parser.add_argument("--norm_wd", action=argparse.BooleanOptionalAction, default=None, help="Weight decay on the 1-D params (RMSNorm weights) too, at the rate of their group. Default: on for muonium with --muonium_rest adamw (its AdamW group decays everything), off for the others (HF Trainer split: matrices + embedding only)")
    parser.add_argument("--muon_momentum", type=float, default=0.95, help="--optimizer muon: Muon Nesterov momentum")
    parser.add_argument("--data_dir", type=str, default=os.path.expandvars("/work/$USER/dolma3_llama"), help="Dir with train.bin and val.bin (uint32, prepare_data.py)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the init, data order and Sophia's label sampling (default: torch's fixed default RNG for the init, Trainer seed 42 for the rest; set it for repeat-seed runs)")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name; also isolates checkpoints in results/<run_name>")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save the final model + tokenizer (default: /work/$USER/llama_models/<run_name or latest>)")
    parser.add_argument("--dyn_every", type=int, default=0, help="Optimizer dynamics (llama_train/optim_dynamics.py): measure per-block gradient / update / curvature / spectral metrics over a 2-step window every this many steps (0 = off). Rank 0 only")
    parser.add_argument("--dyn_spectral_every", type=int, default=None, help="--dyn_every: spectral and effective-rank metrics every this many steps (a multiple of --dyn_every; default --dyn_every, 0 = never)")
    parser.add_argument("--dyn_erank", action=argparse.BooleanOptionalAction, default=True, help="--dyn_every: effective rank of the update (full SVD of each matrix on spectral steps)")
    parser.add_argument("--dyn_track_delta", action=argparse.BooleanOptionalAction, default=True, help="--dyn_every: also the effective rank of the accumulated change W_t - W_(t - dyn_every) (one more copy of the matrices)")
    parser.add_argument("--dyn_out", type=str, default=None, help="--dyn_every: JSONL output (default <save_dir>/optim_dynamics.jsonl)")
    parser.add_argument("--dyn_noise", action=argparse.BooleanOptionalAction, default=False, help="--dyn_every: also the gradient noise scale every --dyn_every steps (GradNoiseScale in llama_train/optim_dynamics.py): each rank draws two fresh micro-batches of the current per-GPU size from train.bin blocks the training never reads (past the end of the ramp sampler's permutation), gradients by autograd.grad (training untouched), sums averaged over ranks. Logs dyn_gns_{G2,trS,B}/<block type>, B in tokens, and dyn_gns_batch/global = the training batch in tokens. Needs --batch_ramp")
    parser.add_argument("--dyn_heatmap_every", type=int, default=10, help="--dyn_every: W&B (layer x block type) heatmap images every this many measurements (0 = none; per-layer values stay in the JSONL)")
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    # Defaults = the best run so far, muonium-normuon0.01-adamw3e-3-ramp-3b (experiments.md 2026-10-16): NorMuon 0.01 for the
    # matrices, AdamW 3e-3 for embedding + norms, wd 0.1, WSD 2% / 10%, batch 32 for the first half of the tokens, then 64
    # (that run switched at a third; the half is from the switch-point sweeps, experiments.md 2026-10-16).
    # LRs tuned for muonium don't carry over to other optimizers, which keep their own defaults.
    if args.learning_rate is None:
        args.learning_rate = 3e-3 if args.optimizer == "muonium" and args.muonium_rest == "adamw" else 6e-4
    if args.muon_lr is None:
        args.muon_lr = 0.01 if args.optimizer == "muonium" else 1.25e-3
    if args.norm_wd is None:  # each optimizer's behavior before the flag existed
        args.norm_wd = args.optimizer == "muonium" and args.muonium_rest == "adamw"
    if args.batch_ramp is None and args.global_batch_size is None:
        args.batch_ramp = "32:0.5,64"
    if args.batch_ramp and args.batch_ramp.lower() == "none":
        args.batch_ramp = None
    if args.global_batch_size is None:
        args.global_batch_size = 64
    return args


def data_dir(args):
    return os.path.expanduser(args.data_dir)


def save_dir(args):
    return os.path.expanduser(args.save_dir or os.path.join(os.path.expandvars("/work/$USER/llama_models"), args.run_name or "latest"))


def print_summary(args):
    print("Model name:", args.model_name, "LR:", args.learning_rate, "Optimizer:", args.optimizer, *(["Muon LR:", args.muon_lr] if args.optimizer in ("muon", "muonium") else []), "WD:", args.weight_decay, "norm WD:", args.norm_wd, "beta2:", args.beta2, "Run name:", args.run_name, "Packing:", args.packing, "Data dir:", data_dir(args))
