import hashlib
import pickle
import torch
import lmdb
import youtokentome as yttm

from pathlib import Path
from torch.utils import data
from tqdm import tqdm


class ByteLevelTextDataset(data.Dataset):
    def __init__(self, path, seq_len, split=b'\n', pad_idx=0, eos_idx=1, cache_dir='.cache'):
        super().__init__()

        self.path = path
        self.seq_len = seq_len

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

        self.vocab_size = 128

        data = Path(path).read_bytes()

        m = hashlib.sha256()
        m.update(data)
        m.update(str((self.seq_len, self.pad_idx, self.eos_idx)).encode())

        if cache_dir is not None:
            cache_path = Path(cache_dir) / (m.hexdigest() + '.p')
            if cache_path.exists():
                print(f'Using cached file {cache_path}')
                self.samples = torch.load(str(cache_path))
                return

        if isinstance(split, str):
            split = split.encode('utf8')
        lines = data.split(split)

        self.samples = torch.full([len(lines), seq_len], fill_value=pad_idx, dtype=torch.long)

        for i, line in enumerate(tqdm(lines)):
            line = line[:seq_len]
            self.samples[i, :len(line)] = torch.tensor(list(line), dtype=torch.long)

        if eos_idx is not None:
            lengths = torch.tensor(list(map(len, lines)), dtype=torch.long).clamp_max(seq_len - 1)
            self.samples[torch.arange(len(lines)), lengths] = eos_idx

        if cache_dir is not None:
            cache_path = Path(cache_dir) / (m.hexdigest() + '.p')
            cache_path.parent.mkdir(exist_ok=True)
            torch.save(self.samples, cache_path)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def seq_to_text(seqs, pad_idx=0, eos_idx=1):
        dim = seqs.dim()

        if dim == 1:
            seqs = seqs.unsqueeze(0)

        if eos_idx is not None:
            lengths = (seqs == eos_idx).type(torch.int).argmax(-1)
        else:
            lengths = torch.full(seqs.size(0), seqs.size(1), dtype=torch.int)

        texts = [''.join(list(map(chr, filter(lambda c: c != pad_idx, seq[:l])))) for l, seq in zip(lengths, seqs)]

        if dim == 1:
            return texts[0]

        return texts

    def to_text(self, seqs):
        return ByteLevelTextDataset.seq_to_text(seqs, self.pad_idx, self.eos_idx)


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
