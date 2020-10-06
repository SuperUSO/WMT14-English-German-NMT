""" Compute error rate.
"""
import torch
import os
import argparse
import utils
import youtokentome as yttm


def main():
    parser = argparse.ArgumentParser(description="Compute BLEU.")
    parser.add_argument('ckpt', type=str, help="Checkpoint to restore.")
    parser.add_argument('--dir', type=str, default="./wmt14", help="Directory of dataset.")
    parser.add_argument('--split', default='test', type=str, help="Specify which split of data to evaluate.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    parser.add_argument('--beams', default=1, type=int, help="Beam Search width.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    # Restore checkpoint
    info = torch.load(args.ckpt)
    cfg = info['cfg']

    # Build model
    bpe_model = yttm.BPE(model=cfg['bpe'])
    model = build_model.Seq2Seq(bpe_model.vocab_size(),
                                bpe_model.vocab_size(),
                                hidden_size=cfg['model']['hidden_size'],
                                encoder_layers=cfg['model']['encoder_layers'],
                                decoder_layers=cfg['model']['decoder_layers'],
                                use_bn=cfg['model']['use_bn'])
    model.load_state_dict(info['weights'])
    model.eval()
    model = model.cuda()

    # Create dataset
    if args.beams == 1:
        batch_size = cfg['train']['batch_size']
    else:
        batch_size = 1
    loader = data.load(args.dir,
                       split=args.split,
                       batch_size=batch_size,
                       bpe_model=bpe_model)

    # Evaluate
    _, bleu = utils.eval_dataset(loader, model, bpe_model, args.beams)
    print ("BLEU on %s set = %.4f" % (args.split, error))


if __name__ == '__main__':
    main()
