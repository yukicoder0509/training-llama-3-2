"""Near-dedup and mild quality filtering of C4 `en` train shards with datatrove, before tokenizing.

--stage dedup: document-level MinHash near-dedup (Lee et al. 2022 style) of the first --num_train_shards
  C4 shards: 5-gram shingles of whitespace words (normalized text), 20 bands x 12 hashes. A pair with Jaccard similarity J is caught with
  probability 1 - (1 - J^12)^20: 0.95 at J=0.85, 0.76 at 0.8, 0.24 at 0.7, 0.04 at 0.6 (threshold ~0.8).
  One doc per duplicate cluster is kept. Output: <out_root>/dedup/*.jsonl.gz.
--stage filter: mild heuristic filters on the deduped docs, to drop clearly unlearnable junk only:
  Gopher repetition (extreme line / paragraph / n-gram repetition, keyword stuffing) and Gopher quality rules
  loosened (< 25 words, < 2 stop words, avg word length outside 3-10, symbol / bullet / ellipsis ratios).
  Output: <out_root>/dedup_filtered/*.jsonl.gz.

Removed docs go to *_removed/ for inspection; per-step counts are in <out_root>/logs/<stage>*/stats.json.
The validation split is never touched (prepare_data.py tokenizes it separately). Tokenize the output with
`prepare_data.py --input_dir <out_root>/dedup[_filtered]`. CPU only; run via dedup_filter.sbatch.
"""

import argparse
import os

import numpy as np

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.text import ngrams, simplify_text


class WhitespaceMinhashSignature(MinhashDedupSignature):
    """MinHash signatures over whitespace tokens of the simplified text (lowercased, punctuation removed), as in
    Lee et al., instead of datatrove's spaCy word tokenizer: spaCy's make_doc was ~57% of the signature time and the
    20-shard job ran at ~17 docs/s per process (505612, ~9.5 h for 7.1M docs)."""

    def get_shingles(self, text):
        words = simplify_text(text, self.config.norm_config).split()
        return np.fromiter(
            (self._hash_func(" ".join(x)) for x in ngrams(words, self.config.n_grams)), dtype=np.uint64
        ).reshape((-1, 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dedup", "filter"], required=True)
    parser.add_argument("--c4_dir", default=os.path.expandvars(
        "/work/$USER/hf_cache/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en"))
    parser.add_argument("--num_train_shards", type=int, default=20, help="First N of 1024 C4 en train shards (~150M tokens each)")
    parser.add_argument("--out_root", default=os.path.expanduser("~/c4_prep"))
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()

    N = args.num_train_shards
    root = args.out_root
    os.makedirs(root, exist_ok=True)
    MINHASH = MinhashConfig(n_grams=5, num_buckets=20, hashes_per_bucket=12)

    if args.stage == "dedup":
        paths_file = os.path.join(root, "c4_train_shards.txt")
        with open(paths_file, "w") as f:
            f.writelines(f"c4-train.{i:05d}-of-01024.json.gz\n" for i in range(N))

        def reader():
            return JsonlReader(args.c4_dir, paths_file=paths_file)

        sigs = LocalPipelineExecutor(
            pipeline=[reader(), WhitespaceMinhashSignature(output_folder=f"{root}/minhash/signatures", config=MINHASH)],
            tasks=N, workers=args.workers, logging_dir=f"{root}/logs/dedup_1_signatures",
        )
        buckets = LocalPipelineExecutor(
            pipeline=[MinhashDedupBuckets(input_folder=f"{root}/minhash/signatures",
                                          output_folder=f"{root}/minhash/buckets", config=MINHASH)],
            tasks=MINHASH.num_buckets, workers=args.workers, logging_dir=f"{root}/logs/dedup_2_buckets", depends=sigs,
        )
        cluster = LocalPipelineExecutor(
            pipeline=[MinhashDedupCluster(input_folder=f"{root}/minhash/buckets",
                                          output_folder=f"{root}/minhash/remove_ids", config=MINHASH, save_cluster_size=True)],
            tasks=1, logging_dir=f"{root}/logs/dedup_3_cluster", depends=buckets,
        )
        dedup = LocalPipelineExecutor(  # same reader and task count as the signature stage: remove_ids are per task
            pipeline=[
                reader(),
                MinhashDedupFilter(input_folder=f"{root}/minhash/remove_ids", load_cluster_sizes=True,
                                   exclusion_writer=JsonlWriter(f"{root}/dedup_removed")),
                JsonlWriter(f"{root}/dedup"),
            ],
            tasks=N, workers=args.workers, logging_dir=f"{root}/logs/dedup_4_filter", depends=cluster,
        )
        dedup.run()
    else:
        LocalPipelineExecutor(
            pipeline=[
                JsonlReader(f"{root}/dedup"),
                GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{root}/filter_removed/repetition")),
                # Mild: < 25 words (spaCy counts punctuation as words) instead of Gopher's 50, and no alpha-word ratio
            # rule (punctuation tokens push normal prose below its 0.8 threshold). On 1 shard (smoke 505522) the
            # defaults removed 22% (alpha 7.7%, < 50 words 7.3%, repetition 6.6%), incl. abstracts and blog posts.
            GopherQualityFilter(min_doc_words=25, max_doc_words=None, max_non_alpha_words_ratio=None,
                                    exclusion_writer=JsonlWriter(f"{root}/filter_removed/quality")),
                JsonlWriter(f"{root}/dedup_filtered"),
            ],
            tasks=N, workers=args.workers, logging_dir=f"{root}/logs/filter",
        ).run()


# datatrove starts workers with forkserver, which re-imports this module: keep the pipeline under main().
if __name__ == "__main__":
    main()
