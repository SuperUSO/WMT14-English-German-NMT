# WMT14 English to German Neural Machine Translation
A Pytorch implementation of NMT using attention-based RNN.

Following [[1]](#References), the sentences are encoded by **BPE (Byte Pair Encoder)** with a shared source-target vocabulary.

## Quick Start
### Install requirements
```
$ pip install -r requirements.txt
```
### Prepare data
Download WMT14 dataset and build the BPE model:
```
python prepare_data.py -h
```
### Train
```
python train.py -h
```
### Evaluate
The default recipe can achieve **~20 BLEU** on the test set after 15 epochs:
```
$ python eval.py exp/default/best.pth --dir ${WMT14} --beams 10 --split dev
Loading DEV dataset ...
DEV set size: 3000
Evaluating: 100%|███████████████████████████████████████████████████████████████████| 3000/3000 [18:59<00:00,  2.63it/s]
BLEU on dev set = 0.2231

$ python eval.py exp/default/best.pth --dir ${WMT14} --beams 10 --split test
Loading TEST dataset ...
TEST set size: 2737
Evaluating: 100%|███████████████████████████████████████████████████████████████████| 2737/2737 [20:12<00:00,  2.26it/s]
BLEU on test set = 0.2098
```

## References
[1] A. Vaswani et al., "Attention Is All You Need".
