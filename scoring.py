"""Shared by score_docs.py and score_bin.py: per-doc mean token loss of a small LM, and the doc metadata cells."""

import numpy as np
import torch
import torch.nn.functional as F


def parse_dir(d):  # data/<source>-<topic>[-<CC quality vigintile>]: same cells as metadata_dist.py
    head, _, tail = d.rpartition("-")
    if tail.isdigit():
        src, _, topic = head.partition("-")
        return src, topic, tail[-2:]
    for src in ("common_crawl", "olmocr_science_pdfs", "stack_edu"):
        if d.startswith(src + "-"):
            return src, d[len(src) + 1:], ""
    return d, "-", ""


def group_name(source, topic):  # the groups of percentiles.csv: cc:<topic>, pdf:<topic>, code:<language>, else the source
    return {"common_crawl": "cc:", "olmocr_science_pdfs": "pdf:", "stack_edu": "code:"}.get(source, "") + (topic if source in ("common_crawl", "olmocr_science_pdfs", "stack_edu") else source)


@torch.no_grad()
def score_stream(model, tokens, doc_start, window=2048, batch_tokens=65536, log=print, tok_loss=None):
    """Per-doc (loss sum, predictions) for docs tokens[doc_start[i]:doc_start[i+1]] (tokens: any int array, e.g. a memmap).

    Each doc is split into <= window-token windows scored on their own (FlashAttention varlen, positions restart per
    window); a window's last token has no target. The windows tile the stream, so a batch is a contiguous slice.
    tok_loss: optional float32 CUDA tensor of doc_start[-1] - doc_start[0] entries; gets the loss of predicting token
    j + 1 at entry j (relative to doc_start[0]); entries without a target are left as they are.
    """
    k = len(doc_start) - 1
    n_tok = np.diff(doc_start)
    w_start = np.concatenate([np.arange(s, e, window) for s, e in zip(doc_start[:-1], doc_start[1:])])
    w_end = np.append(w_start[1:], doc_start[-1])
    w_doc = np.repeat(np.arange(k), -(-n_tok // window))
    assert len(w_doc) == len(w_start)
    W = model.lm_head.weight
    loss_sum = torch.zeros(k, dtype=torch.float64, device="cuda")
    pred = torch.zeros(k, dtype=torch.float64, device="cuda")
    b0, nb, total = 0, 0, doc_start[-1] - doc_start[0]
    while b0 < len(w_start):
        b1 = int(np.searchsorted(w_start, w_start[b0] + batch_tokens, side="right"))  # windows starting within budget
        b1 = max(b0 + 1, b1 - 1 if w_end[b1 - 1] - w_start[b0] > batch_tokens else b1)
        a, e = int(w_start[b0]), int(w_end[b1 - 1])
        cu = torch.from_numpy(np.append(w_start[b0:b1], e) - a).to(torch.int32).cuda()
        lens = torch.diff(cu)
        pos = torch.arange(e - a, device="cuda") - torch.repeat_interleave(cu[:-1], lens)
        ids = torch.from_numpy(np.asarray(tokens[a:e]).astype(np.int64)).cuda()
        tok_doc = torch.repeat_interleave(torch.from_numpy(w_doc[b0:b1]).cuda(), lens)
        h = model.model(input_ids=ids[None], position_ids=pos[None], cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                        max_length_q=window, max_length_k=window).last_hidden_state[0]
        valid = torch.ones(e - a, dtype=torch.bool, device="cuda")
        valid[cu[1:] - 1] = False
        tgt = torch.cat([ids[1:], ids[:1]])
        for c in range(0, e - a, 8192):
            l = F.cross_entropy((h[c:c + 8192] @ W.T).float(), tgt[c:c + 8192], reduction="none")
            v, d = valid[c:c + 8192], tok_doc[c:c + 8192]
            loss_sum.index_add_(0, d[v], l[v].double())
            pred.index_add_(0, d[v], torch.ones_like(l[v], dtype=torch.float64))
            if tok_loss is not None:
                o = a - int(doc_start[0]) + c
                tok_loss[o:o + len(l)][v] = l[v]
        b0, nb = b1, nb + 1
        if nb % 1000 == 0:
            log(f"batch {nb}: {(e - doc_start[0]) / total:.1%} of tokens")
    return loss_sum.cpu().numpy(), pred.cpu().numpy()
