import argparse
import pickle
import lmdb
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from vq_text_gan.datasets import BPEDataset
from vq_text_gan.utils import get_default_device


def extract_codes(args):
    device = get_default_device(args.device)

    print('Loading model')
    model = torch.load(args.ckpt).to(device)

    print('Loading data')
    dataset = BPEDataset(args.dataset)

    save_path = args.save_path
    if not save_path:
        save_path = 'codes'

    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(save_path, map_size=map_size)

    batches = DataLoader(dataset, batch_size=args.batch_size)

    with torch.no_grad():
        model.eval()

        with env.begin(write=True) as txn:
            index = 0

            for batch in tqdm(batches):
                batch = batch.to(device)

                codes, _, _ = model.encode(batch)

                sample_wise_codes = list(zip(*[code.cpu().numpy() for code in codes]))

                for sample_codes in sample_wise_codes:
                    txn.put(str(index).encode('utf-8'), pickle.dumps(sample_codes))
                    index += 1

            txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    extract_codes(args)
