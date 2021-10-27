# PhysioSSL: A Python Toolbox for Physiological Time-series Representation Learning

<div align=center><img src="https://i.loli.net/2021/10/03/32AUGkZcoeORWij.png" width=25% height=25%/></div>

## Introduction

## Installation

Install from PyPi:

```shell
pip install physiossl
```

or install via the GitHub link:

```shell
pip install git+https://github.com/larryshaw0079/PhysioSSL
```

## Getting Started

## Implemented Algorithms

| Algo        | Title | Year | Ref          |
| ----------- | ----- | ---- | ------------ |
| TCL         |       |      | [[1]](#ref1) |
| RP          |       |      | [[2]](#ref2) |
| TS          |       |      | [[2]](#ref2) |
| CPC         |       |      |              |
| Moco        |       |      |              |
| SimCLR      |       |      |              |
| DPC         |       |      |              |
| DPCM        |       |      |              |
| TripletLoss |       |      |              |
| DCC         |       |      |              |
| TNC         |       |      |              |
| TSTCC       |       |      |              |
| CoSleep     |       | 2021 |              |

## Supported Datasets

### Sleep Stage Classification

### Emotion Recognition

### Human Activity Recognition

## Citing

```latex
@misc{qfxiao2021physiossl,
  author =       {Qinfeng Xiao},
  title =        {PhysioSSL: A Python Toolbox for Physiological Time-series Representation Learning},
  howpublished = {\url{https://github.com/larryshaw0079/PhysioSSL}},
  year =         {2021}
}
```

## Reference

> <div id="ref1">
> [1] Hyvarinen, Aapo and Morioka, Hiroshi,. (2016). Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA. Advances in Neural Information Processing Systems.
> </div>
> <div id="ref2">[2] Banville, H., Chehab, O., Hyvarinen, A., Engemann, D., & Gramfort, A. (2020). Uncovering the structure of clinical EEG signals with self-supervised learning. Journal of neural engineering, 10.1088/1741-2552/abca18. Advance online publication. https://doi.org/10.1088/1741-2552/abca18</div>
