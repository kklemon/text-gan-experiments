#!/usr/bin/env python
import lmdb
import logging
import pickle
import torch
import config

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from vq_vae_text.datasets import ByteLevelTextDataset, BPEDataset, OneBillionBenchmarkBPE
from vq_vae_text.datasets import CodeRow


def extract_codes(device, args):
    logging.info('Loading model')
    model = torch.load(args.ckpt).to(device)

    logging.info('Loading data')

    # dataset_builder = config.datasets.get(args.dataset)
    # if not dataset_builder:
    #     raise Exception(f'Unknown dataset \'{args.dataset}\'. Valid options are {str(list(config.datasets.keys()))}')

    #dataset =
    dataset = OneBillionBenchmarkBPE('/home/kris/data/text/1-billion-word-language-modeling-benchmark-r13output/train_250k_10b.p')

    save_path = args.save_path
    if not save_path:
        save_path = 'codes'

    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(save_path, map_size=map_size)

    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    with torch.no_grad():
        model.eval()

        with env.begin(write=True) as txn:
            index = 0

            for batch in tqdm(batches):
                batch = batch.to(device)

                _, z, diff, codes = model(batch)

                texts = batch.cpu().numpy()
                codes = list(zip(*[code.cpu().numpy() for code in codes]))

                for text, code in zip(texts, codes):
                    row = CodeRow(text=text, codes=code)
                    txn.put(str(index).encode('utf-8'), pickle.dumps(row))

                    index += 1

            txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))
