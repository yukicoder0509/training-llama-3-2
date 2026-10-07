# Lab 5: Optimizing Token Efficiency for Llama 3.2

🪧 Courses: Optimizers (https://app.notion.com/p/Optimizers-3dcaca4a7ce280508392f41b5a9082b4?pvs=21), Hyperparameter Tuning (https://app.notion.com/p/Hyperparameter-Tuning-3dcaca4a7ce28029863dfd8911cd6f8e?pvs=21)

## Description

The choice of optimizer affects how efficiently a language model learns from its training data. AdamW [1] adapts updates to individual parameters, while Muon [2] transforms matrix updates through approximate orthogonalization. In this lab, you are to explore optimizer design by training a model from scratch using the Llama 3.2 1B architecture [3] on the `allenai/dolma3_mix-150B-1025` dataset [4].

Your goal is to get the lowest perplexity after training on no more than 6B tokens, with a context window of 8192 tokens. To this end, you are allocated at most 64 H200-hours worth of compute per run, with at most 8 H200s used in each run. The token budget counts non-padding tokens processed during training, including repeated tokens each time they are processed.

To construct the training and evaluation sets, shuffle the dataset’s `train` split with seed 42 and reserve the last 50,000 documents for evaluation. These documents must be excluded from training. Perform this split before tokenizing or packing documents into training sequences.

This lab focuses on token-efficient training. You may experiment with update rules, momentum, weight decay, learning rates, schedules, and the assignment of different optimizers to different parameter groups. You are recommended to establish an AdamW baseline before experimenting with Muon or your own modifications. 

In addition, to conserve compute and simulate real world experiment planning, each *pair* is limited to 3x 64 H200-hour runs. Although you may circumvent this rule by running numerous 63 hour runs, the spirit of this rule is to encourage careful planning towards experiment scaling. Since we expect real-world pretraining to cost ~8400 H200-hours per run, it is infeasible to test every hypothesis at a full pretraining scale. 

You may plan your experiments in a ladder-like fashion, in which you quickly test a series of optimizer/hyperparameter designs in ~1-2 hour runs, then test their scaling by a few ~8-16 hour runs, and finally commit a recipe to test out in your 64 hour runs.

When planning how many GPUs to use in parallel, please remember that *batch size is a hyperparameter too*. This may further assist you https://sadhikamalladi.github.io/blog/2024/01/22/SDEs-ScalingRules/.

Your training should start from randomly initialized weights. You may reuse your training code from Lab 4, or use HF’s stock implementations. You are encouraged to read Llama 3’s technical report for pretraining practice reference [5]. 

You are expected to log your training metrics to wandb under the `lab5-training-llama` project. To facilitate ease of run comparisons across peers, you are required to log to following metrics to wandb. Each metric’s value is logged “on that step” unless otherwise specified (e.g., if logging steps is 10, you should log the tok/s on step 10 instead of the average tok/s of steps 1 - 10). 

- `train/loss` : The average loss in the duration of `logging_steps`
- `train/grad_norm`
- `train/learning_rate`
- `train/tokens_per_second`
- `train/total_tokens_seen` : The amount of tokens the model has seen from the beginning of training up to the logging step
- `eval/perplexity` : The model’s perplexity on a separate, held out evaluation set not included in your training process (you may use up to 20% of the OJ’s hold out eval set).
    - This should be done no less frequently than once every 10% of training steps (e.g., if you estimate training would be 10k steps, you should do it no less often than once every 1k steps).

Upon completion of your training, upload your model weights and tokenizer to HuggingFace. Then, in your OJ script, change the variable `eval_model_id`. The OJ will pull the model and evaluate its perplexity on the reserved 50,000 documents. Include your training configuration and wandb run link so that the optimizer settings, token budget, and training duration can be verified.

## Deadlines

- You must achieve the baseline of ppl 20 before 2026年10月9日, this should be feasible with a 16 H200-hours run.
- The full lab is due 2026年10月20日

## References

[1] https://arxiv.org/abs/1711.05101
[2] https://arxiv.org/abs/2502.16982
[3] https://huggingface.co/meta-llama/Llama-3.2-1B
[4] https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025
[5] https://arxiv.org/abs/2407.21783