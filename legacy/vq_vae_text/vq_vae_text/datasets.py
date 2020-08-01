import os
import lmdb
import pickle
import torch
import hashlib
import youtokentome as yttm

from pathlib import Path
from functools import partial
from collections import namedtuple
from torch.utils import data
from torchtext.data import Dataset, Field, Example
from bpemb import BPEmb
from tqdm import tqdm



class Seq2SeqDataset(Dataset):
    def __init__(self,
                 path,
                 fields=None,
                 tokenizer=list,
                 batch_first=True,
                 lower=False,
                 split_by='\n',
                 fix_length=None,
                 pad_token='﹏',
                 unk_token='�',
                 sos_token='◁',
                 eos_token='▷',
                 **kwargs):
        if not fields:
            self.SRC = Field(tokenize=tokenizer,
                             lower=lower,
                             pad_token=pad_token,
                             unk_token=unk_token,
                             eos_token=eos_token,
                             batch_first=batch_first,
                             fix_length=fix_length)
            self.TRG = Field(tokenize=tokenizer,
                             lower=lower,
                             pad_token=pad_token,
                             unk_token=unk_token,
                             init_token=sos_token,
                             batch_first=batch_first,
                             fix_length=fix_length)
            build_vocab = True
        else:
            self.SRC, self.TRG = fields
            build_vocab = False

        fields = [('src', self.SRC),
                  ('trg', self.TRG)]

        with open(path) as fp:
            lines = fp.read().split(split_by)

        make_example = partial(Example.fromlist, fields=fields)
        examples = list(map(make_example, zip(lines, lines)))

        super().__init__(examples, fields)

        if build_vocab:
            self.build_shared_vocab()
        else:
            self.vocab = self.SRC.vocab
            assert self.vocab == self.SRC.vocab == self.TRG.vocab

    def build_shared_vocab(self):
        self.SRC.build_vocab()
        self.TRG.build_vocab()

        self.vocab = self.SRC.vocab
        self.vocab.extend(self.TRG.vocab)

        self.SRC.build_vocab(self)
        self.TRG.build_vocab(self)

        self.vocab.extend(self.SRC.vocab)
        self.vocab.extend(self.TRG.vocab)

        self.SRC.vocab = self.vocab
        self.TRG.vocab = self.vocab

        assert self.vocab == self.SRC.vocab == self.TRG.vocab

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None, test=None, **kwargs):
        if not train:
            raise Exception('At least train must be provided.')

        train_data = cls(os.path.join(path, train), **kwargs)
        fields = train_data.SRC, train_data.TRG
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


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
    def __init__(self, path, seq_len, bpe_model, pad_idx=0, eos_idx=3):
        super().__init__()

        self.path = path
        self.seq_len = seq_len

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

        samples = Path(path).read_text().split('\n')

        self.bpe = yttm.BPE(bpe_model)
        self.vocab_size = self.bpe.vocab_size()

        encoded = self.bpe.encode(samples, eos=True)

        self.lengths = list(map(lambda x: min(x, seq_len), map(len, encoded)))

        self.samples = torch.full([len(encoded), seq_len], fill_value=pad_idx, dtype=torch.long)
        for i, sample in enumerate(encoded):
            self.samples[i, :self.lengths[i]] = torch.tensor(sample[:self.lengths[i]])

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def seq_to_text(self, seqs):
        dim = seqs.dim()

        if dim == 1:
            seqs = seqs.unsqueeze(0)

        if False: #self.eos_idx is not None:
            lengths = torch.max((seqs == self.eos_idx).type(torch.int), dim=-1)[1]
        else:
            lengths = torch.full((seqs.size(0),), seqs.size(1), dtype=torch.int)

        texts = self.bpe.decode([sample[:length] for sample, length in zip(seqs.detach().cpu().numpy().tolist(), lengths)])

        if dim == 1:
            return texts[0]

        return texts


class OneBillionBenchmarkBPE(data.Dataset):
    def __init__(self, path):
        super().__init__()

        data = pickle.loads(Path(path).read_bytes())
        samples = data['samples']

        self.seq_length = data['seq_length']
        self.bpe = yttm.BPE(data['bpe_path'])
        self.vocab = self.bpe.vocab()
        self.vocab_size = len(self.vocab)
        self.lengths = list(map(len, samples))

        self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx = list(range(4))

        self.samples = torch.full([len(samples), self.seq_length], fill_value=self.pad_idx, dtype=torch.long)
        for i, sample in enumerate(tqdm(samples)):
            self.samples[i, :self.lengths[i]] = torch.tensor(sample[:self.lengths[i]])

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def seq_to_text(self, seqs):
        dim = seqs.dim()

        if dim == 1:
            seqs = seqs.unsqueeze(0)

        if False: #self.eos_idx is not None:
            lengths = torch.max((seqs == self.eos_idx).type(torch.int), dim=-1)[1]
        else:
            lengths = torch.full((seqs.size(0),), seqs.size(1), dtype=torch.int)

        texts = self.bpe.decode([sample[:length] for sample, length in zip(seqs.detach().cpu().numpy().tolist(), lengths)])

        if dim == 1:
            return texts[0]

        return texts


def create_byte_level_spiegel_splits(
        data_dir,
        seq_length,
        train_file='train.txt',
        valid_file='val.txt',
        pad_idx=0,
        eos_idx=1
):
    builder = partial(ByteLevelTextDataset, seq_len=seq_length, pad_idx=pad_idx, eos_idx=eos_idx)
    data_dir = Path(data_dir)
    train_data = builder(str(data_dir / train_file))
    valid_data = builder(str(data_dir / valid_file))
    return train_data, valid_data


def create_bpe_dataset_splits(
        data_dir,
        seq_length,
        bpe_model,
        train_file='train.txt',
        valid_file='val.txt',
):
    builder = partial(BPEDataset, bpe_model=bpe_model, seq_len=seq_length)
    data_dir = Path(data_dir)
    train_data = builder(str(data_dir / train_file))
    valid_data = builder(str(data_dir / valid_file))
    return train_data, valid_data


def create_one_billion_word_bpe_splits(train_file, valid_file):
    train_data = OneBillionBenchmarkBPE(train_file)
    valid_data = OneBillionBenchmarkBPE(valid_file)
    return train_data, valid_data


CodeRow = namedtuple('CodeRow', ['text', 'codes'])


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

            row = pickle.loads(txn.get(key))

        return CodeRow(
            text=row.text,
            codes=[torch.from_numpy(code) for code in row.codes]
        )

