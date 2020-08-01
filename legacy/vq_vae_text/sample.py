import argparse
import random
import torch
import torch.nn.functional as F

from vq_vae_text.datasets import ByteLevelTextDataset


def sample_top(model, sos_idx, sample_length, num_samples, device):
    output = torch.empty([sample_length + 1, num_samples], dtype=torch.long, device=device)
    output[:1, :] = sos_idx

    for i in range(sample_length):
        logits = model(output[:i + 1])
        probas = F.softmax(logits, dim=-1)[-1]
        samples = torch.multinomial(probas, num_samples=1)
        #samples = probas.argmax(-1)
        samples = samples.view(probas.size(0))
        output[i + 1, :] = samples

    return output[1:]


def sample_bottom(model, top_samples, sos_idx, sample_length, num_samples, device):
    output = torch.empty([sample_length + 1, num_samples], dtype=torch.long, device=device)
    output[:1, :] = sos_idx

    for i in range(sample_length):
        logits = model(top_samples, output[:i + 1])
        probas = F.softmax(logits, dim=-1)[-1]
        samples = torch.multinomial(probas, num_samples=1)
        #samples = probas.argmax(-1)
        samples = samples.view(probas.size(0))
        output[i + 1, :] = samples

    return output[1:]


def decode_code(model, codes):
    code_a, code_b = codes

    #return model.decode_code(codes)

    quant_a = model.quantize_a.embed_code(code_a)
    #quant_a = quant_a.view(quant_a.size(0), quant_a.size(1) // model.quantize_a.d_slice, -1)
    quant_a = quant_a.permute(0, 2, 1)

    quant_b = model.quantize_b.embed_code(code_b)
    #quant_b = quant_b.view(quant_b.size(0), quant_b.size(1) // model.quantize_b.d_slice, -1)
    quant_b = quant_b.permute(0, 2, 1)

    dec = model.decode((quant_a, quant_b))

    return dec


def decode(decoder, bottom, top):
    decoded = decode_code(decoder, (bottom, top)).argmax(-1)
    return ByteLevelTextDataset.seq_to_text(decoded)


def header(text):
    print('*' * 8 + ' ' + text + ' ' + '*' * 8)


def prefix_token(data, token: int):
    new_data = torch.empty(data.size(0) + 1, data.size(1), dtype=data.dtype, device=data.device)
    new_data[0, :] = token
    new_data[1:, :] = data
    return new_data


@torch.no_grad()
def main(args):
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sos_idx = args.num_vq_embeds

    top_model = torch.load(args.top_ckpt).to(device)
    top_model.eval()

    bottom_model = torch.load(args.bottom_ckpt).to(device)
    bottom_model.eval()

    decoder = torch.load(args.ae_ckpt).to(device)
    decoder.eval()

    top_samples = sample_top(top_model, sos_idx, args.top_length, args.num_samples, device)
    bottom_samples = sample_bottom(bottom_model, top_samples, sos_idx, args.bottom_length, args.num_samples, device)

    header('Random samples: ')
    print('\n'.join(decode(decoder, bottom_samples.T, top_samples.T)))

    if args.dataset:
        if not args.dataset_seq_length:
            print('--dataset-seq-length must be provided if dataset is given')

        dataset = ByteLevelTextDataset(args.dataset, args.dataset_seq_length)
        random_texts = torch.stack(random.choices(dataset, k=args.num_samples)).to(device)

        _, _, (real_bottom, real_top) = decoder.encode(random_texts)
        reals = decode(decoder, real_bottom, real_top)

        print()
        header('Sampling with reconstructed top latents')
        recon_top = top_model(prefix_token(real_top.T, sos_idx)).argmax(-1)[:-1].T
        decoded = decode(decoder, real_bottom, recon_top)
        print('\n\n'.join('\n'.join(pair) for pair in zip(reals, decoded)))

        print()
        header('Sampling with reconstructed bottom latents')
        recon_bottom = bottom_model(real_top.T, prefix_token(real_bottom.T, sos_idx)).argmax(-1)[:-1].T
        decoded = decode(decoder, recon_bottom, real_top)
        print('\n\n'.join('\n'.join(pair) for pair in zip(reals, decoded)))

        print()
        header('Sampling with ground truth top latents')
        bottom_samples = sample_bottom(bottom_model, real_top.T, sos_idx, args.bottom_length, args.num_samples, device)
        decoded = decode(decoder, bottom_samples.T, real_top)
        print('\n\n'.join('\n'.join(pair) for pair in zip(reals, decoded)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--top-ckpt', type=str, required=True)
    parser.add_argument('--bottom-ckpt', type=str, required=True)
    parser.add_argument('--ae-ckpt', type=str, required=True)
    parser.add_argument('--num-vq-embeds', type=int, required=True)
    parser.add_argument('--top-length', type=int, required=True)
    parser.add_argument('--bottom-length', type=int, required=True)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset-seq-length', type=int)
    args = parser.parse_args()

    main(args)
