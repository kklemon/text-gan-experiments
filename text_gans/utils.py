import math
import logging
import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from collections import Counter


def get_default_device(default=None):
    if default:
        device = default
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def setup_run(run_name, results_dir='results', create_dirs=('checkpoints', 'samples')):
    results_dir = Path(results_dir)

    run_dir = results_dir / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'-{run_name}')
    run_dir.mkdir(parents=True)

    for d in create_dirs:
        (run_dir / d).mkdir()

    return run_dir


def setup_logging(log_file):
    logger = logging.getLogger()

    if logger.handlers:
        logger.removeHandler(logger.handlers[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=str(log_file))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)


def compute_entropy(counts):
    length = sum(counts.values())
    probs = list(map(lambda x: x / length, counts.values()))
    return -sum([p * math.log2(p) for p in probs if p])


def update_average(model_tgt, model_src, beta):
    # utility function for toggling the gradient requirements of the nets
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


def repr_list(l):
    return ', '.join(f'{v:.2f}' for v in l)


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.orthogonal_(m.weight)


def apply_spectral_norm(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.utils.spectral_norm(m)


def get_log2_seq_length(seq_length):
    log2_seq_length = math.log2(seq_length)
    if log2_seq_length != int(log2_seq_length):
        raise ValueError(f'Sequence length must be a power of two. Got {seq_length}.')

    log2_seq_length = int(log2_seq_length)
    if log2_seq_length <= 1:
        raise ValueError(f'Sequence length must be a power of two and >= 4. Got {seq_length}.')

    return log2_seq_length
