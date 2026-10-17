"""Train the Llama 3.2 1B architecture on Dolma 3 with the HF Trainer (see README; launched by run.sbatch).

The pieces live in llama_train/: cli (flags, defaults), plan (steps, batch ramp, LR schedule, checkpoint steps), model,
data, optim, callbacks, trainer. main() puts them together.
"""

import json
import os
import time

import torch
from transformers import AutoTokenizer, TrainingArguments

from llama_train import cli
from llama_train import plan as P
from llama_train.callbacks import (DeadlineCallback, ProfilerCallback, SaveAtTokensCallback, SophiaHessianCallback,
                                   StepTimerCallback, WandbTokenAxisCallback, slurm_time_left)
from llama_train.data import DocPackingCollate, load_datasets
from llama_train.model import TOKENIZER, build_model, flops_per_token, llama_config
from llama_train.optim import build_optimizer, uses_sophia
from llama_train.trainer import CustomTrainer


def main(argv=None):
    start_time = time.time()
    print(torch.__version__, "Cuda:", torch.cuda.is_available(), torch.version.cuda)  # GPU check

    args = cli.parse_args(argv)
    cli.print_summary(args)
    save_dir = cli.save_dir(args)
    plan = P.make_plan(args, int(os.environ.get("WORLD_SIZE", 1)))
    lr_lambda = P.lr_lambda(args, plan)
    block_size = plan.block_size

    print("=== Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = llama_config(args)
    model = build_model(args, config)

    print("=== Loading Dolma 3 tokens...")
    train_ds, val_ds, dev_ds = load_datasets(cli.data_dir(args), block_size)
    # One pass at most: the budget must not repeat tokens unintentionally (repeats would count against it anyway)
    assert plan.tokens_seen(plan.max_steps) <= len(train_ds) * block_size, "train.bin has fewer tokens than the budget; raise --train_frac in prepare_data.py"

    ckpt_dir = P.ckpt_dir(args)
    save_steps = P.save_steps(args, plan, ckpt_dir)

    sched_type, sched_kwargs, warmup_steps = P.hf_schedule(args, plan)
    training_args = TrainingArguments(
        output_dir=ckpt_dir,  # only checkpoints (--save_at_tokens) are written here
        run_name=args.run_name,
        per_device_train_batch_size=plan.per_device_batch,
        per_device_eval_batch_size=plan.per_device_batch,
        gradient_accumulation_steps=plan.grad_accum,
        bf16=True,
        num_train_epochs=1,
        max_steps=plan.max_steps,
        max_grad_norm=1.0,

        # Logging, eval and reporting
        logging_steps=10,
        report_to="wandb",  # Log to W&B (project from WANDB_PROJECT, see run.sbatch)
        eval_strategy="steps" if args.eval else "no",
        eval_steps=plan.eval_steps,
        save_strategy="no",

        # DDP (see --ddp_* flags; the bf16 comm hook is set in CustomTrainer)
        ddp_find_unused_parameters=args.ddp_find_unused,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,

        # Optimizer
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=args.beta2,
        adam_epsilon=args.adam_eps,
        weight_decay=args.weight_decay,

        # Scheduler (plan.hf_schedule; wsd2 and --schedule_unit tokens replace it with lr_lambda in CustomTrainer)
        lr_scheduler_type=sched_type,
        lr_scheduler_kwargs=sched_kwargs,
        warmup_steps=warmup_steps,
        seed=42 if args.seed is None else args.seed,  # data order (and RNG after Trainer init)
    )
    print("=== Training arguments: ", training_args)

    optimizer, sophia_opt = build_optimizer(args, model, config, plan, training_args)

    step_timer = StepTimerCallback(plan, flops_per_token(model, config, block_size), start_time)
    sophia_stats = {}
    sophia_cb = SophiaHessianCallback(args, plan, model, optimizer, sophia_opt, sophia_stats) if uses_sophia(args) else None
    time_left = slurm_time_left()
    deadline = start_time + time_left - 300 if time_left else float("inf")  # leave 5 min for the final eval + save
    print("Deadline:", f"{(deadline - start_time) / 3600:.2f} h from start" if time_left else "none")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DocPackingCollate(config.bos_token_id, block_size) if args.packing == "doc" else None,
        # Sophia's Hessian pass before step_timer, so its time counts in tokens_per_second / MFU
        callbacks=([sophia_cb] if sophia_cb else [])
        + [DeadlineCallback(deadline, start_time), step_timer, WandbTokenAxisCallback(args, plan)]
        + ([SaveAtTokensCallback(save_steps, ckpt_dir, plan)] if save_steps else [])
        + ([ProfilerCallback(args.profile_dir, args.profile_start, args.profile_cpu)] if args.profile_dir else []),
        optimizers=(optimizer, None),  # None -> Trainer builds AdamW from args; scheduler always from args
        cli_args=args, plan=plan, lr_lambda=lr_lambda, dev_dataset=dev_ds, step_timer=step_timer, extra_logs=sophia_stats,
        keep_step_batches=sophia_cb is not None,
    )
    if sophia_cb:
        sophia_cb.trainer = trainer

    print("=== Starting training...")
    if args.resume_from:
        print(f"=== Resuming from {args.resume_from}")
    trainer.train(resume_from_checkpoint=args.resume_from)
    if args.eval and trainer.state.global_step % plan.eval_steps:  # final eval unless the last step just ran one
        trainer.evaluate()
    budget_completed = trainer.state.global_step >= plan.max_steps
    print(f"=== Trained on {plan.tokens_seen(trainer.state.global_step):,} tokens in {trainer.state.global_step}/{plan.max_steps} steps"
          + ("" if budget_completed else " -- WARNING: token budget NOT completed (deadline)"))
    if trainer.is_world_process_zero():
        import wandb
        if wandb.run is not None:  # filter / check runs by this in W&B before comparing them
            wandb.run.summary.update({"budget_completed": budget_completed, "final_tokens_seen": plan.tokens_seen(trainer.state.global_step)})
    trainer.save_model(save_dir)  # main process only; config + generation_config + safetensors
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(save_dir)  # the OJ pulls the tokenizer from the same repo
        # transformers 5 writes RoPE as "rope_parameters"; 4.x would ignore it and fall back to rope_theta 10000.
        # Add the 4.x keys too so the model evaluates the same with either version.
        cfg_path = os.path.join(save_dir, "config.json")
        cfg = json.load(open(cfg_path))
        rope = dict(cfg.get("rope_parameters") or {})
        cfg.setdefault("rope_theta", rope.pop("rope_theta", 500000.0))
        cfg.setdefault("rope_scaling", rope or None)
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"=== Training finished. Saved to {save_dir} at {time.time() - start_time:.0f}s. Push with:")
    print(f"    python push_model.py --model_dir {save_dir} --repo_id {args.model_name}")


if __name__ == "__main__":
    main()
