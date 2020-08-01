import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from vq_vae_text.datasets import LMDBDataset
from vq_vae_text.models import SHARNN, TransformerEncoderModel, TransformerDecoderModel
from vq_vae_text.utils import setup_logging_from_args, save_checkpoint

from functools import partial
from pytorch_lamb import Lamb
from torchtext import data
from torchtext.datasets import WikiText103, WikiText2


print = logging.info


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def process_sample(sample, sos_idx, eos_idx):
    src = torch.empty(len(sample) + 1, dtype=sample.dtype, device=sample.device)
    trg = torch.empty(len(sample) + 1, dtype=sample.dtype, device=sample.device)

    src[0] = sos_idx
    src[1:] = sample

    trg[-1] = eos_idx
    trg[:-1] = sample

    return src, trg


class Data(torch.utils.data.Dataset):
    def __init__(self, dataset, stage, transform_func):
        self.dataset = dataset
        self.stage = stage
        self.transform_func = transform_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        code_row = self.dataset[idx]

        if self.stage == 'bottom':
            return self.transform_func(code_row.codes[0]), code_row.codes[1]
        elif self.stage == 'top':
            return code_row.codes[0], self.transform_func(code_row.codes[1])
        else:
            raise ValueError(f'No stage {self.stage}')


def compute_accuracy(pred, truth):
    return (pred == truth).type(torch.float).mean()


def prepare_batch(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device).T.contiguous()
    return [prepare_batch(datum, device) for datum in data]


def train(model, optimizer, criterion, device, dataloader, stage, tau, epoch, scheduler=None, debug=False):
    total_loss = 0
    total_acc = 0

    model.train()

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch = prepare_batch(batch, device)
        bottom, top = batch

        if stage == 'top':
            top_src, top_trg = top
            obj = top_trg

            logits = model(top_src)
        else:
            btt_src, btt_trg = bottom
            obj = btt_trg

            logits = model(top, btt_src)

        logits /= tau
        logits = F.log_softmax(logits, dim=-1)

        loss = criterion(logits.view(-1, logits.size(-1)), obj.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        total_loss += loss.data
        total_acc += compute_accuracy(logits.argmax(-1).view(-1), obj.view(-1))

        if step % args.log_interval == 0 and step > 0:
            cur_loss = total_loss / args.log_interval
            cur_acc = total_acc / args.log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                  'loss {:5.4f} | ppl {:8.2f} | bpc {:8.3f} | acc {:05.2f}'.format(
                epoch, step, len(dataloader), optimizer.param_groups[0]['lr'],
                cur_loss, math.exp(cur_loss),
                cur_loss / math.log(2), cur_acc * 100.0))
            total_loss = 0
            total_acc = 0

        if scheduler is not None:
            scheduler.step()

        if debug:
            break


def evaluate(model, criterion, device, dataloader, stage, tau, epoch):
    total_loss = 0
    total_acc = 0

    model.eval()

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = prepare_batch(batch, device)
            bottom, top = batch

            if stage == 'top':
                top_src, top_trg = top
                obj = top_trg

                logits = model(top_src)
            else:
                btt_src, btt_trg = bottom
                obj = btt_trg

                logits = model(top, btt_src)

            logits /= tau
            logits = F.log_softmax(logits, dim=-1)

            loss = criterion(logits.view(-1, logits.size(-1)), obj.view(-1))

            acc = compute_accuracy(logits.argmax(-1).view(-1), obj.view(-1))

            total_loss += loss.data
            total_acc += acc

    loss = total_loss / len(dataloader)
    acc = total_acc / len(dataloader)

    print('| epoch {:3d} | loss {:5.4f} | ppl {:8.2f} | bpc {:8.3f} | acc {:05.2f}'.format(
        epoch, loss, math.exp(loss), loss / math.log(2), acc * 100.0
    ))

    return loss


def load_data(path, batch_size, stage, transform_func, is_train=True):
    raw_data = LMDBDataset(path)
    data = Data(raw_data, stage, transform_func)

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=is_train)


def main(args):
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.model = f'{args.model}_{args.stage}'

    save_path = setup_logging_from_args(args)

    sos_idx = args.num_vq_embeds
    eos_idx = sos_idx + 1

    transform_func = partial(process_sample, sos_idx=sos_idx, eos_idx=eos_idx)

    train_data = load_data(args.train_data, args.batch_size, args.stage, transform_func, is_train=True)
    val_data = load_data(args.val_data, args.batch_size, args.stage, transform_func)

    vocab_size = args.num_vq_embeds + 2

    print(f'Training {args.stage} stage')
    if args.stage == 'top':
        model = TransformerEncoderModel(vocab_size, 256, 4, 512, 4, dropout=0.2).to(device)
    else:
        model = TransformerDecoderModel(vocab_size, 256, 4, 512, 2, 4, dropout=0.2).to(device)

    model.train()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=args.wdecay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # from pytorch_lamb import Lamb
    # optimizer = Lamb(model.parameters(), min_trust=0.25)

    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_data), gamma=0.9)

    for epoch in range(args.epochs):
        train(model, optimizer, criterion, device, train_data, args.stage, args.tau, epoch, schedule, args.debug)

        logging.info('Saving model')
        save_checkpoint(model, epoch, save_path)

        print('Evaluating')
        evaluate(model, criterion, device, val_data, args.stage, args.tau, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--device', type=str)
    parser.add_argument('--model-args', type=str)
    parser.add_argument('--ae-model-checkpoint', type=str)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, required=True)
    parser.add_argument('--num-vq-embeds', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--wdecay', type=float, default=1.2e-6)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Learning rate decay gamma.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--stage', choices=['bottom', 'top'], required=True)
    args = parser.parse_args()

    main(args)
