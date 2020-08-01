import pickle
import torch
import lmdb
import youtokentome as yttm

from pathlib import Path
from torch.utils import data


class BPEDataset(data.Dataset):
    def __init__(self, path):
        super().__init__()

        data = pickle.loads(Path(path).read_bytes())

        self.samples = data['samples']
        self.seq_length = data['seq_length']
        self.bpe = yttm.BPE(data['bpe_path'])
        self.vocab = self.bpe.vocab()
        self.vocab_size = len(self.vocab)

        self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx = list(range(4))

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = torch.full([self.seq_length], self.pad_idx, dtype=torch.long)
        item[:len(s)] = torch.tensor(s, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.samples)

    def seq_to_text(self, seqs):
        dim = seqs.dim()

        if dim == 1:
            seqs = seqs.unsqueeze(0)

        if self.eos_idx is not None:
            eos_mask = (seqs == self.eos_idx).type(torch.int)
            lengths = torch.max(eos_mask, dim=-1)[1]
            lengths.masked_fill_(eos_mask.sum(1) == 0, self.seq_length)
        else:
            lengths = torch.full([seqs.size(0)], seqs.size(1), dtype=torch.int)

        texts = self.bpe.decode([sample[:length] for sample, length in zip(seqs.detach().cpu().numpy().tolist(), lengths)])

        if dim == 1:
            return texts[0]

        return texts


class LMDBDataset(data.Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            codes = pickle.loads(txn.get(key))

        return [torch.from_numpy(code) for code in codes]
