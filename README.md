# ANTNet in PyTorch
This repository contains a PyTorch implementation of the mobile convolutional 
architecture given in [ANTNets](https://arxiv.org/abs/1904.03775).

## Implementation comparison
The following model comparison is with reference to ANTNet for ImageNet as 
given in the paper.

| Parameter | Official | Ours |
| ---------- | -------- | ----- |
| Number of parameters (g = 1) | 3.7M | 3.81M |
| Number of parameters (g = 2) | 3.2M | 3.33M |

## Usage 
The `cfg_antnet.py` module has the model and training configuration in the form
of `dict`. This module can be passed to a training module to initialize the
model and kickstart training. An example module `train.py` is given for reference
and can be used as follows:
```python
python3 train.py --cfg cfg_antnet.py
```

## Contribution
Please feel free to write an issue in case you find a bug. Additional recommendations
for features are welcome too!
