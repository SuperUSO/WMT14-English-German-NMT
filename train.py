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


def get_lr(optimizer):
    """
    A helper function to get the solver's learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_history(save_path, message):
    """
    A helper function to log the history.
    The history text file is saved as: ${SAVE_PATH}/history.csv

    Args:
        save_path (string): The location to log the history.
        message (string): The message to log.
    """
    fname = os.path.join(save_path,'history.csv')
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write("datetime,epoch,learning rate,train loss,dev loss,BLEU\n")
            f.write("%s\n" % message)
    else:
        with open(fname, 'a') as f:
            f.write("%s\n" % message)


def save_checkpoint(filename, save_path, epoch, dev_bleu, cfg, model, optimizer, scheduler):
    filename = os.path.join(save_path, filename)
    info = {'epoch': epoch,
            'dev_bleu': dev_bleu,
            'cfg': cfg,
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
    torch.save(info, filename)


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('cfg', type=str, help="Specify which experiment config file to use.")
    parser.add_argument('--dir', type=str, default="./wmt14", help="Directory of dataset.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    parser.add_argument('--workers', default=0, type=int, help="How many subprocesses to use for data loading.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_path = os.path.splitext(args.cfg)[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)

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
    train_loader = data.load(args.dir,
                             split='train',
                             batch_size=cfg['train']['batch_size'],
                             bpe_model=bpe_model,
                             workers=args.workers)
    dev_loader = data.load(args.dir,
                           split='dev',
                           batch_size=cfg['train']['batch_size'],
                           bpe_model=bpe_model)

    # Training criteria
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['init_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=cfg['train']['decay_factor'],
                                                           patience=cfg['train']['patience'],
                                                           threshold=0.01,
                                                           min_lr=1e-6)
    assert cfg['train']['metric'] in ['loss', 'bleu']

    # Restore checkpoints
    if os.path.exists(os.path.join(save_path, 'last.pth')):
        info = torch.load(os.path.join(save_path, 'last.pth'))
        epoch = info['epoch']
        model.load_state_dict(info['weights'])
        optimizer.load_state_dict(info['optimizer'])
        scheduler.load_state_dict(info['scheduler'])
    else:
        epoch = 0

    if os.path.exists(os.path.join(save_path, 'best.pth')):
        info = torch.load(os.path.join(save_path, 'best.pth'))
        best_epoch = info['epoch']
        best_bleu = info['dev_bleu']
    else:
        best_epoch = 0
        best_bleu = 0

    while (1):
        print ("---")
        epoch += 1
        print ("Epoch: %d" % (epoch))
        # Show learning rate
        lr = get_lr(optimizer)
        print ("Learning rate: %f" % lr)

        # Training loop
        model.train()
        train_loss = []
        train_tqdm = tqdm(train_loader, desc="Training")
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
            # Save best model
            save_checkpoint("best.pth", save_path, best_epoch, best_bleu, cfg, model, optimizer, scheduler)
        print ("Best dev. BLEU: %.4f @epoch: %d" % (best_bleu, best_epoch))

        # Update learning rate scheduler
        if cfg['train']['metric'] == 'loss':
            scheduler.step(dev_loss)
        else:
            scheduler.step(1-dev_bleu)

        # Save checkpoint
        save_checkpoint("last.pth", save_path, epoch, dev_bleu, cfg, model, optimizer, scheduler)

        # Logging
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msg = "%s,%d,%f,%f,%f,%f" % (datetime, epoch, lr, np.mean(train_loss), dev_loss, dev_bleu)
        log_history(save_path, msg)


if __name__ == '__main__':
    main()
