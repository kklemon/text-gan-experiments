from vq_vae_text.models import *
from vq_vae_text.datasets import (
    create_bpe_dataset_splits,
    create_byte_level_spiegel_splits,
    create_one_billion_word_bpe_splits
)


datasets = {
    'byte_level_spiegel': create_byte_level_spiegel_splits,
    'bpe_spiegel': create_bpe_dataset_splits,
    'one_billion_word_bpe': create_one_billion_word_bpe_splits
}


model_names = {
    'textcnn': TextCNN,
    'textcnnV2': TextCNNV2,
    'textcnnV2attn': TextCNNV2Attn,
    'textcnn_attn_v2': TextCNNVAttnV2,
    'self_attn_ae': SelfAttentionAE,
    'textcnn_adain': TextCNNAdaIN,
    'textcnn_vq2': TextCNNVQ2,
    'textcnn_vq3': TextCNNVQ3
}


model_configs = {
    TextCNN: dict(
        vocab_size=128,
        channel=128,
        res_channel=32,
        n_res_block=2,
        tau=1.0,
        pad_idx=0,
        input_embed_dim=64,
        num_vq_embeds=128,
        vq_embeds_dim=4
    ),
    TextCNNV2: dict(
        channel=256,
        res_channel=64,
        n_res_blocks=4,
        n_encoders=1,
        tau=0.1,
        pad_idx=0,
        vq_loss_alpha=0.25,
        num_vq_embeds=512,
        vq_embeds_dim=16,
    ),
    TextCNNV2Attn: dict(
        vocab_size=128,
        embed_dim=128,
        channel=256,
        res_channel=64,
        n_res_block=2,
        tau=1.0,
        pad_idx=0,
        eos_idx=1,
        vq_loss_alpha=0.25,
        num_vq_embeds=2 ** 10,
        vq_embed_dim=8,
        input_noise=0.05,
        d_slice=1
    ),
    TextCNNVAttnV2: dict(
        vocab_size=128,
        channel=256,
        n_fold=4,
        tau=0.1,
        pad_idx=0,
        num_vq_embeds=2**5,
        vq_embeds_dim=128,
        d_slice=2
    ),
    SelfAttentionAE: dict(
        vocab_size=128,
        dim=256,
        dim_feedforward=1024,
        n_fold=4,
        tau=1.0,
        pad_idx=0,
        num_vq_embeds=2**10,
        vq_embeds_dim=128,
        #d_slice=1
    ),
    TextCNNAdaIN: dict(
        vocab_size=128,
        channel=256,
        n_fold=4,
        tau=1.0,
        pad_idx=0,
        num_vq_embeds=2**8,
        vq_embeds_dim=64,
        vq_loss_alpha=0.25
    ),
    TextCNNVQ2: dict(
        embed_dim=256,
        channel=256,
        res_channel=256,
        n_res_block=2,
        tau=0.1,
        pad_idx=0,
        vq_loss_alpha=0.25,
        num_vq_embeds=2**12,
        vq_embed_dim=8,
        input_noise=0.05,
        d_slice=1
    ),
    TextCNNVQ3: dict(
        embed_dim=128,
        channel=64,
        res_channel=64,
        n_res_block=2,
        tau=0.1,
        pad_idx=0,
        vq_loss_alpha=0.25,
        config=[
            (2, 2 ** 12),
            (1, 2 ** 11),
            (1, 2 ** 10),
        ],
        attn=True,
        vq_embed_dim=4,
        input_noise=0.05,
        d_slice=1
    )
}