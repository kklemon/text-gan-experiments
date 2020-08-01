import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from vq_vae_text.models import SHARNN, TransformerEncoderModel

from pytorch_lamb import Lamb
from torchtext import data
from torchtext.datasets import WikiText103, WikiText2


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def prepare_batch(batch, sos_idx, eos_idx):
    ext_size = (batch.size(0) + 1, batch.size(1))
    src = torch.empty(ext_size, dtype=batch.dtype, device=batch.device)
    trg = torch.empty(ext_size, dtype=batch.dtype, device=batch.device)

    src[0, :] = sos_idx
    src[1:, :] = batch

    trg[-1, :] = eos_idx
    trg[:-1, :] = batch

    return src, trg


class Data(torch.utils.data.Dataset):
    def __init__(self, dataset, stage, transform_f):
        self.dataset = dataset
        self.stage = stage
        self.transform_f = transform_f

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        code_row = self.dataset[idx]

        if self.stage == 'bottom':
            return self.transform_f(code_row.codes[0]), code_row.codes[1]
        elif self.stage == 'top':
            return self.transform_f(code_row.codes[1])
        else:
            raise ValueError(f'No stage {self.stage}')


def main(args):
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text_field = data.Field(tokenize=list)
    datasets = WikiText2.splits(text_field)
    text_field.build_vocab(datasets[0])

    train_iter, test_iter, val_iter = data.BPTTIterator.splits(datasets, batch_size=32, bptt_len=512, device=device)

    vocab = text_field.vocab

    print(f'Vocab size: {len(vocab)}')

    model_args = dict(
        rnn_type='lstm',
        ntoken=args.num_latents,
        ninp=256,
        nhid=1024,
        nlayers=2
    )
    if args.model_args:
        model_args.update(dict(eval(args.model_args)))

    model = SHARNN(**model_args).to(device)
    model.train()

    criterion = nn.NLLLoss()

    #optim = torch.optim.SGD(model.parameters(), lr=5.0)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(10):
        hidden = None
        mems = None

        total_loss = 0

        for step, batch in enumerate(train_iter):
            optim.zero_grad()

            if hidden is not None:
                hidden = repackage_hidden(hidden)
            if mems is not None:
                mems = repackage_hidden(mems)

            output, hidden, mems, attn_outs, _ = model(batch.text, hidden, return_h=True, mems=mems)

            logits = model.decoder(output)
            logits = F.log_softmax(logits, dim=-1)

            assert logits.size(1) == batch.target.size(1)

            loss = criterion(logits.view(-1, logits.size(-1)), batch.target.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optim.step()

            total_loss += loss.data

            if step % args.log_interval == 0 and step > 0:
                cur_loss = total_loss / args.log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                      'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, step, len(train_iter), optim.param_groups[0]['lr'],
                                  cur_loss, math.exp(cur_loss),
                                  cur_loss / math.log(2)))
                total_loss = 0

    # model = TransformerEncoderModel(len(vocab), 64, 2, 1024, 2).to(device)
    # model.train()
    #
    # criterion = nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    # schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_iter), gamma=0.9)
    #
    # for epoch in range(10):
    #     total_loss = 0
    #
    #     for step, batch in enumerate(train_iter):
    #         optimizer.zero_grad()
    #
    #         logits = model(batch.text)
    #         logits = F.log_softmax(logits, dim=-1)
    #
    #         assert logits.size(1) == batch.target.size(1)
    #
    #         loss = criterion(logits.view(-1, logits.size(-1)), batch.target.view(-1))
    #         loss.backward()
    #
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #
    #         optimizer.step()
    #
    #         total_loss += loss.data
    #
    #         if step % args.log_interval == 0 and step > 0:
    #             cur_loss = total_loss / args.log_interval
    #             print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
    #                   'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
    #                 epoch, step, len(train_iter), optimizer.param_groups[0]['lr'],
    #                               cur_loss, math.exp(cur_loss),
    #                               cur_loss / math.log(2)))
    #             total_loss = 0
    #
    #         schedule.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--device', type=str)
    parser.add_argument('--model-args', type=str)
    parser.add_argument('--ae-model-checkpoint', type=str)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str)
    parser.add_argument('--num-latents', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Learning rate decay gamma.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--stage', choices=['bottom', 'top'])
    args = parser.parse_args()

    main(args)
