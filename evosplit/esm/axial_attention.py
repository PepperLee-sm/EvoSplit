# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn


class RowSelfAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        row_att_all=False,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        attn_weights_all = []
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights_ = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, start : start + max_rows]
                if self_attn_padding_mask is not None
                else None,
                row_att_all=row_att_all,
            )
            if row_att_all:
                attn_weights, attn_weights_part = attn_weights_
                attn_weights_all.append(attn_weights_part)
            else:
                attn_weights = attn_weights_
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(x[start : start + max_rows], attn_probs)
            outputs.append(output)

        output = torch.cat(outputs, 0)
        if row_att_all:
            attn_weights_all = torch.cat([attn_weights for attn_weights in attn_weights_all], dim=0)
            attn_probs_all = attn_weights_all.softmax(-1)
            return output, attn_probs, attn_probs_all
        else:
            return output, attn_probs

    def compute_attention_weights(
        self,
        x,
        scaling: float,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        row_att_all=False,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(q)
        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k) # hnij

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )
        if row_att_all:
            attn_weights_all = torch.einsum(f"rinhd,rjnhd->rhnij", q/scaling*self.scaling, k).cpu() # hnij
            return attn_weights, attn_weights_all
        else:
            return attn_weights 

    def compute_attention_update(
        self,
        x,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        include_row=None,
        row_att_all=False
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if include_row is not None:
            scaling = self.align_scaling(x[include_row].unsqueeze(0))
            if row_att_all:
                attn_weights, attn_weights_all = self.compute_attention_weights(
                    x[include_row].unsqueeze(0), scaling, self_attn_mask, self_attn_padding_mask, row_att_all=row_att_all,
                )
                attn_probs_all = attn_weights_all.float().softmax(-1)
            else:
                attn_weights = self.compute_attention_weights(
                    x[include_row].unsqueeze(0), scaling, self_attn_mask, self_attn_padding_mask,
                )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            if row_att_all:
                return output, attn_probs, attn_probs_all
            else:
                return output, attn_probs
        else:
            if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
                return self._batched_forward(x, self_attn_mask, self_attn_padding_mask, row_att_all)
            else:
                scaling = self.align_scaling(x)
                if row_att_all:
                    attn_weights, attn_weights_all = self.compute_attention_weights(
                        x, scaling, self_attn_mask, self_attn_padding_mask, row_att_all=row_att_all,
                    )
                    attn_probs_all = attn_weights_all.float().softmax(-1)
                else:
                    attn_weights = self.compute_attention_weights(
                        x, scaling, self_attn_mask, self_attn_padding_mask
                    )
                attn_probs = attn_weights.softmax(-1)
                attn_probs = self.dropout_module(attn_probs)
                output = self.compute_attention_update(x, attn_probs)
                if row_att_all:
                    return output, attn_probs, attn_probs_all
                else:
                    return output, attn_probs


class ColumnSelfAttention(nn.Module):
    """Compute self-attention over columns of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        # Updated by Shimian Li
        outputs = torch.empty_like(x)
        attns = torch.empty(self.num_heads, num_cols, batch_size, num_rows, num_rows)
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, :, start : start + max_cols]
                if self_attn_padding_mask is not None
                else None,
            )
            # outputs.append(output)
            outputs[:, start:start+max_cols, :, :] = output
            attns[:, start:start+max_cols, :, :, :] = attn
        return outputs, attns

    def compute_attention_update(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with padding
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=x.device,
                dtype=x.dtype,
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            q *= self.scaling

            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (num_rows * num_cols) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                x,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
            return self.compute_attention_update(x, self_attn_mask, self_attn_padding_mask)
