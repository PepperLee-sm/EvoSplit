# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from ..modules import (
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
)

from ..axial_attention import RowSelfAttention, ColumnSelfAttention



class MSATransformer(nn.Module):
    @classmethod
    def add_args(cls, parser):
        # fmt: off
        parser.add_argument(
            "--num_layers",
            default=12,
            type=int,
            metavar="N",
            help="number of layers"
        )
        parser.add_argument(
            "--embed_dim",
            default=768,
            type=int,
            metavar="N",
            help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias",
            action="store_true",
            help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=3072,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=12,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--attention_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--activation_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--max_tokens_per_msa",
            default=2 ** 14,
            type=int,
            help=(
                "Used during inference to batch attention computations in a single "
                "forward pass. This allows increased input sizes with less memory."
            ),
        )
        # fmt: on

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )

        if getattr(self.args, "embed_positions_msa", False):
            emb_dim = getattr(self.args, "embed_positions_msa_dim", self.args.embed_dim)
            self.msa_position_embedding = nn.Parameter(
                0.01 * torch.randn(1, 1024, 1, emb_dim),
                requires_grad=True,
            )
        else:
            self.register_parameter("msa_position_embedding", None)

        self.dropout_module = nn.Dropout(self.args.dropout)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    self.args.dropout,
                    self.args.attention_dropout,
                    self.args.activation_dropout,
                    getattr(self.args, "max_tokens_per_msa", self.args.max_tokens)
                )
                for _ in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions,
            self.args.embed_dim,
            self.padding_idx,
        )
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False, include_row=None, row_att_all=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(tokens)
        x += self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x += self.msa_position_embedding[:, :num_alignments]

        x = self.emb_layer_norm_before(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            row_attn_weights = []
            # col_attn_weights = []
            # col_attn_weights = torch.empty((batch_size, self.num_layers, self.args.attention_heads, seqlen, num_alignments, num_alignments)) # B x L x H x C x R x R
            # row_attn_weights = torch.empty((batch_size, self.num_layers, self.args.attention_heads, seqlen, seqlen)) # row_attentions: B x L x H x C x C

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            if (layer_idx+1 == self.num_layers) and row_att_all:
                x = layer(
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_head_weights=need_head_weights,
                        include_row=include_row,
                        row_att_all=True,
                    )
                
            else:
                x = layer(
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_head_weights=need_head_weights,
                        include_row=include_row,
                    )
            if need_head_weights:
                if len(x) == 3:
                    x, col_attn, row_attn = x
                elif len(x) == 4:
                    x, col_attn, row_attn, row_attn_all = x
                    row_attn_weights_all = row_attn_all.permute(0, 2, 1, 3, 4) # last layer
                # H x C x B x R x R -> B x H x C x R x R
                # col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
                # H x B x C x C -> B x H x C x C
                row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            # import psutil
            # print(psutil.Process().memory_info().rss / (1024 * 1024))
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)
        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # # col_attentions: B x L x H x C x R x R
            # col_attentions = torch.stack(col_attn_weights, 1)
            # row_attentions: B x L x H x C x C 
            row_attentions = torch.stack(row_attn_weights, 1)
            # row_attentions_all = torch.stack(row_attn_weights_all, 2)
            # result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if row_att_all:
                result["row_attentions_all"] = row_attn_weights_all
            if return_contacts:
                # result["contacts_all"] = []
                # for i in range(row_attentions_all.size()[0]):
                #     contact = self.contact_head(tokens[0, i, :].unsqueeze(0).unsqueeze(0), row_attentions_all[i])
                #     result["contacts_all"].append(contact)
                contacts = self.contact_head(tokens, row_attentions)
                result["contacts"] = contacts
        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers

    def max_tokens_per_msa_(self, value: int) -> None:
        """The MSA Transformer automatically batches attention computations when
        gradients are disabled to allow you to pass in larger MSAs at test time than
        you can fit in GPU memory. By default this occurs when more than 2^14 tokens
        are passed in the input MSA. You can set this value to infinity to disable
        this behavior.
        """
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value
