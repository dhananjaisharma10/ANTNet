import torch
from importlib import import_module
from argparse import ArgumentParser

from antnet import AntNet


def argparser():
    parser = ArgumentParser(description='Parser for ANTNet')
    parser.add_argument('--cfg', required=True,
                        help='Configuration file for the model')
    return parser.parse_args()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = argparser()
    filename = args.cfg[:-3]
    cfg = import_module(filename)
    model_cfg = cfg.model
    model = AntNet(**model_cfg)
    print('Number of params:', count_params(model))
    inp = torch.randn((32, 3, 224, 224))
    out = model(inp)
    print(out.size())
