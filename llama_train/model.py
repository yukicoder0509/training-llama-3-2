"""The Llama 3.2 1B model: config, random init, Liger fused LM head + loss, per-layer torch.compile."""

import json

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, set_seed

TOKENIZER = "NousResearch/Llama-3.2-1B"  # tokenizer.json byte-identical to meta-llama/Llama-3.2-1B (gated for us)


def llama_config(args):
    # Llama 3.2 1B architecture (meta-llama/Llama-3.2-1B config.json), randomly initialized: N(0, 0.02) Linear/embedding
    # weights (LlamaPreTrainedModel._init_weights), tied embeddings, GQA 32 query / 8 KV heads, rope_theta 5e5 with the
    # llama3 RoPE scaling. Written out instead of loaded since the meta-llama repo is gated for us.
    # The 128,256 vocab is already a multiple of 128 (fast LM-head GEMMs), so no vocab padding is needed.
    config = LlamaConfig(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, head_dim=64, hidden_act="silu",
        max_position_embeddings=131072, rms_norm_eps=1e-5, initializer_range=0.02, tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                      "original_max_position_embeddings": 8192},
        attention_bias=False, attention_dropout=0.0, mlp_bias=False,
        bos_token_id=128000, eos_token_id=128001, pad_token_id=128004, use_cache=False,
    )
    if args.config_overrides:
        overrides = json.loads(args.config_overrides)
        if "rope_scaling" in overrides and overrides["rope_scaling"] is None:
            # transformers 5 keeps RoPE in rope_parameters (rope_theta included); None there breaks the model, so
            # "no scaling" = plain RoPE with the same theta
            overrides.pop("rope_scaling")
            overrides["rope_parameters"] = {"rope_type": "default", "rope_theta": config.rope_parameters["rope_theta"]}
        config.update(overrides)
        print("Config overrides:", args.config_overrides)
    return config


def build_model(args, config):
    if args.seed is not None:
        set_seed(args.seed)  # the init draws from the global RNG; without --seed it is torch's fixed default seed
    assert args.packing == "concat" or args.attn_implementation.startswith("flash_attention"), "--packing doc needs FlashAttention varlen (--attn_implementation flash_attention_3)"
    model = AutoModelForCausalLM.from_config(config, attn_implementation=args.attn_implementation)
    print("Attention implementation:", model.config._attn_implementation)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B")
    if args.fused_ce:
        patch_fused_ce(model)
    if args.torch_compile:
        compile_layers(model, args)
    return model


def patch_fused_ce(model):
    """Fused LM head + cross-entropy (Liger): computes the logits, loss and their gradients chunk by chunk inside
    the forward, instead of HF's full (tokens x vocab) logits upcast to fp32 for log_softmax. Same shift and
    normalization as HF's ForCausalLMLoss: sum / num_items_in_batch when the Trainer passes it (grad accum), else mean.
    Patched on the instance so the saved config stays LlamaForCausalLM."""
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast
    fused_ce = {r: LigerFusedLinearCrossEntropyLoss(reduction=r) for r in ("mean", "sum")}

    def fused_ce_forward(input_ids=None, labels=None, num_items_in_batch=None, sample_labels=False, **kwargs):
        if "cu_seq_lens_q" in kwargs:  # --packing doc: the document count varies per batch; one dynamic-shape compile, not one per count
            torch._dynamo.maybe_mark_dynamic(kwargs["cu_seq_lens_q"], 0)
        hidden = model.model(input_ids=input_ids, **kwargs).last_hidden_state
        if sample_labels:  # Sophia's Hessian estimate: mean CE against labels drawn from the model's own predictions
            return CausalLMOutputWithPast(loss=sampled_label_loss(hidden))
        if labels is None:
            return CausalLMOutputWithPast(logits=model.lm_head(hidden))
        hidden = hidden[:, :-1].reshape(-1, hidden.size(-1))  # tokens < n predict n
        targets = labels[:, 1:].reshape(-1)
        if num_items_in_batch is None:
            loss = fused_ce["mean"](model.lm_head.weight, hidden, targets)
        else:
            loss = fused_ce["sum"](model.lm_head.weight, hidden, targets) / num_items_in_batch
        return CausalLMOutputWithPast(loss=loss)

    def sampled_label_loss(hidden, chunk=2048):
        """CE of every position against y ~ softmax(logits) (official train_sophiag.py), without the full logits:
        exact categorical samples by Gumbel-max (argmax of logits - log Exp(1)) chunk by chunk."""
        h = hidden.reshape(-1, hidden.size(-1))
        w = model.lm_head.weight
        with torch.no_grad():
            samples = []
            for c in h.split(chunk):
                logits = (c @ w.t()).float()
                samples.append(logits.sub_(torch.empty_like(logits).exponential_().log_()).argmax(-1))
        return fused_ce["mean"](w, h, torch.cat(samples))

    model.forward = fused_ce_forward


def compile_layers(model, args):
    """Compile only the decoder layers: the memory-bound RMSNorm / cast / SwiGLU / elementwise kernels, while the LM head +
    loss stays in Liger's kernels outside the compiled graph (compiling the whole body made GPT-2's attention 2.7x slower)."""
    if args.batch_ramp:  # a static recompile per batch size, not slower dynamic-shape kernels after the first change
        torch._dynamo.config.automatic_dynamic_shapes = False
    if args.attn_implementation.startswith("flash_attention"):
        # The FlashAttention Hub kernel is not traceable (graph break at the call), and the frame resumed after it would
        # recompile per layer_idx (static int attribute) and fall back to eager past the recompile limit (8): make
        # it dynamic. Also skip HF's warning_once that it casts autocast's fp32 q/k/v to bf16 (another graph break).
        import logging
        torch._dynamo.config.ignore_logging_functions.add(logging.Logger.warning_once)
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.recompile_limit = 32
    for layer in model.model.layers:
        layer.compile()


def flops_per_token(model, config, block_size):
    """Model FLOPs per token (fwd + bwd): 6 * params + attention 12 * n_layer * hidden * seq_len (PaLM/nanoGPT MFU formula)."""
    return 6 * sum(p.numel() for p in model.parameters()) + 12 * config.num_hidden_layers * config.hidden_size * block_size
