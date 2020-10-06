import yaml
import os
import re
import argparse
import time
import torch
import utils
import numpy as np
import youtokentome as yttm
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser(description="Train the model on DEVELOPMENT set to make sure it can overfit.")
    parser.add_argument('cfg', type=str, help="Specify which experiment config file to use.")
    parser.add_argument('--dir', type=str, default="./wmt14", help="Directory of dataset.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Build model
    bpe_model = yttm.BPE(model=cfg['bpe'])
    model = build_model.Seq2Seq(bpe_model.vocab_size(),
                                bpe_model.vocab_size(),
                                hidden_size=cfg['model']['hidden_size'],
                                encoder_layers=cfg['model']['encoder_layers'],
                                decoder_layers=cfg['model']['decoder_layers'],
                                drop_p=cfg['model']['drop_p'],
                                use_bn=cfg['model']['use_bn'])
    model = model.cuda()

    # Create dataset
    dev_loader = data.load(args.dir,
                           split='dev',
                           batch_size=32,
                           bpe_model=bpe_model)

    # Training criteria
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['init_lr'])

    epoch = 0
    best_epoch = 0
    best_bleu = 0
    while (1):
        print ("---")
        epoch += 1
        print ("Epoch: %d" % (epoch))

        # Training loop
        model.train()
        train_loss = []
        train_tqdm = tqdm(dev_loader, desc="Training")
        for (xs, ys) in train_tqdm:
            loss = model(xs.cuda(), ys.cuda())
            train_loss.append(loss.item())
            train_tqdm.set_postfix(loss="%.3f" % np.mean(train_loss))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)   # Gradient clipping
            optimizer.step()

        # Validation loop
        model.eval()
        dev_loss, dev_bleu = utils.eval_dataset(dev_loader, model, bpe_model)
        print ("Dev. loss: %.3f," % dev_loss, end=' ')
        print ("dev. BLEU: %.4f" % dev_bleu)
        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            best_epoch = epoch
        print ("Best dev. BLEU: %.4f @epoch: %d" % (best_bleu, best_epoch))


if __name__ == '__main__':
    main()
