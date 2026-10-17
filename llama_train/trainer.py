"""The HF Trainer with train.py's LR schedules, batch-ramp sampler, dev-set eval, bf16 DDP hook and lab metrics."""

import math

import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from .plan import RampBatchSampler


class CustomTrainer(Trainer):
    step_batches, _batches_step = [], -1

    def __init__(self, *a, cli_args, plan, lr_lambda=None, dev_dataset=None, step_timer=None, extra_logs=None,
                 keep_step_batches=False, **kw):
        """cli_args: train.py flags. lr_lambda: LR factor per step (wsd2 / --schedule_unit tokens), else the Trainer's own
        scheduler. dev_dataset: evaluated next to eval_dataset as eval_dev_*. step_timer: StepTimerCallback for
        tokens_per_second / mfu. extra_logs: dict merged into train logs (Sophia's win rate). keep_step_batches: keep each
        optimizer step's micro-batches (Sophia's Hessian pass)."""
        # set before Trainer.__init__, which calls create_accelerator_and_postprocess
        self.cli_args, self.plan, self.lr_lambda, self.dev_dataset = cli_args, plan, lr_lambda, dev_dataset
        self.step_timer, self.extra_logs, self.keep_step_batches = step_timer, extra_logs or {}, keep_step_batches
        super().__init__(*a, **kw)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.keep_step_batches:  # keep this optimizer step's micro-batches for the Hessian estimate
            if self._batches_step != self.state.global_step:
                self.step_batches, self._batches_step = [], self.state.global_step
            self.step_batches.append(inputs)
        return super().training_step(model, inputs, num_items_in_batch)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if self.cli_args.ddp_bf16_grads and self.accelerator.ddp_handler is not None:
            from accelerate.utils import DDPCommunicationHookType
            self.accelerator.ddp_handler.comm_hook = DDPCommunicationHookType.BF16  # registered when DDP wraps the model

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_lambda is not None and self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, self.lr_lambda)
            self._created_lr_scheduler = True
        return super().create_scheduler(num_training_steps, optimizer)

    def get_train_dataloader(self):
        p = self.plan
        if p.ramp is None:
            return super().get_train_dataloader()
        sampler = RampBatchSampler(len(self.train_dataset), p.ramp, p.world_size, self.args.seed, p.grad_accum)
        return self.accelerator.prepare(DataLoader(self.train_dataset, batch_sampler=sampler,
                                                   collate_fn=self.data_collator, pin_memory=True))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.dev_dataset is not None and eval_dataset is None and metric_key_prefix == "eval":
            metrics.update(super().evaluate(self.dev_dataset, ignore_keys, "eval_dev"))
        return metrics

    # Lab metrics. WandbCallback prefixes train logs with "train/" and "eval_*" with "eval/": loss (mean over the
    # logging interval), grad_norm and learning_rate come from the Trainer; tokens_per_second (of the logging step
    # itself), total_tokens_seen and eval_perplexity are added here.
    def log(self, logs, *args, **kwargs):
        tokens_seen = self.plan.tokens_seen
        if "loss" in logs:
            logs["ppl"] = math.exp(logs["loss"])
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)
            if self.step_timer is not None and self.step_timer.times:
                logs["tokens_per_second"], logs["mfu"] = self.step_timer.last_step_throughput()
            logs.update(self.extra_logs)
        if "eval_loss" in logs:
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)  # x-axis for eval/perplexity vs tokens
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])
        if "eval_dev_loss" in logs:
            logs["total_tokens_seen"] = tokens_seen(self.state.global_step)
            logs["eval_dev_perplexity"] = math.exp(logs["eval_dev_loss"])
        super().log(logs, *args, **kwargs)
