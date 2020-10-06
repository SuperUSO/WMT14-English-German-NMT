import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu


def decode(s_in_batch, bpe_model):
    """
    Args:
        s_in_batch (list(list(integer))): A mini-batch of sentences in subword IDs.

    Returns:
        s_out_batch (list(list(string))): A mini-batch of sentences in words.
    """
    s_out_batch = []
    for s in s_in_batch:
        s_out = []
        for id in s:
            if id == 3: # <s>
                continue
            elif id == 2: # </s>
                break
            s_out.append(id)
        s_out_batch.append(s_out)
    s_out_batch = bpe_model.decode(s_out_batch)   # list(string)
    s_out_batch = [s.split() for s in s_out_batch]   # list(list(string))
    return s_out_batch


def eval_dataset(dataloader, seq2seq, bpe_model, beam_width=1, max_size=10000):
    """
    Calculate loss and BLEU score on a corpus.

    Args:
        seq2seq (nn.Module): Neural machine translation model.
        bpe_model (youtokentome.BPE): Byte-pair encoding model.
        max_size (integer): Maximum number of sentences to evaluate. This is to prevent OOM issue.
    """
    total_loss = []
    preds_corpus = []
    gt_corpus = []
    with torch.no_grad():
        eval_tqdm = tqdm(dataloader, desc="Evaluating")
        for (xs, ys) in eval_tqdm:
            if len(preds_corpus) > max_size:
                break

            total_loss.append(seq2seq(xs.cuda(), ys.cuda()).item())

            preds_batch, _ = seq2seq(xs.cuda(), beam_width=beam_width)   # torch.LongTensor, [batch_size, seq_length]
            preds_batch = decode(preds_batch.tolist(), bpe_model)        # list(list(integer))
            preds_corpus += preds_batch                                  # list(list(integer))

            gt_batch = decode(ys.tolist(), bpe_model)   # list(list(integer))
            gt_corpus += gt_batch                       # list(list(integer))

    loss = np.mean(total_loss)
    gt_corpus = [[gt] for gt in gt_corpus]   # corpus_bleu() requires gt_corpus to be list(list(list(integer)))
    bleu_score = corpus_bleu(gt_corpus, preds_corpus)
    return loss, bleu_score
