"""
Attention mechanisms for sequence-to-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor [batch, ..., seq_len_q, d_k]
        K: Key tensor [batch, ..., seq_len_k, d_k]
        V: Value tensor [batch, ..., seq_len_v, d_k]
        mask: Optional mask [batch, ..., seq_len_q, seq_len_k]
              Values: 1 for positions to attend, 0 for positions to mask

    Returns:
        output: Attention output [batch, ..., seq_len_q, d_k]
        attention_weights: Attention weights [batch, ..., seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)

    # TODO: Compute attention scores
    # TODO: Scale scores
    # TODO: Apply mask if provided (use masked_fill to set masked positions to -inf)
    # TODO: Apply softmax
    # TODO: Apply attention to values

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask != None:
        m = mask
        if m is not None:
            if m.dim() == 2:  # padding mask [batch, seq]
                m = m[:, None, None, :]  # [batch, 1, 1, seq]
            elif m.dim() == 3:  # [batch, seq_q, seq_k]
                m = m[:, None, :, :]  # [batch, 1, seq_q, seq_k]
            elif m.dim() == 4:
                pass  # already [batch, heads, seq_q, seq_k]
            
            # expand singleton dims
            batch_size, seq_q, seq_k = scores.shape[0], scores.shape[2], scores.shape[3]
            if m.size(0) == 1:
                m = m.expand(batch_size, -1, -1, -1)
            if m.size(1) == 1:
                m = m.expand(-1, scores.size(1), -1, -1)
            
            # ensure mask matches seq_q, seq_k
            if m.size(2) != seq_q or m.size(3) != seq_k:
                m = m.expand(-1, -1, seq_q, seq_k)
            
            m = m.to(dtype=torch.bool, device=scores.device)
            
            scores = scores.masked_fill(~m, float('-inf'))
            #scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits d_model into num_heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Initialize linear projections for Q, K, V
        # TODO: Initialize output projection

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        # TODO: Reshape and transpose to split heads

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor with shape [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, d_k = x.size()

        # TODO: Transpose and reshape to combine heads

        batch_size, num_heads, seq_len, d_k = x.size()
        
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # TODO: Linear projections
        # TODO: Split heads
        # TODO: Apply scaled dot-product attention
        # TODO: Combine heads
        # TODO: Apply output projection

        Q = self.split_heads(self.Q(query))
        K = self.split_heads(self.K(key))
        V = self.split_heads(self.V(value))

        atten_out, atten_weight = scaled_dot_product_attention(Q, K, V, mask)

        combined_output = self.combine_heads(atten_out)

        output = self.out(combined_output)

        return output, atten_weight


def create_causal_mask(seq_len, device=None):
    """
    Create causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] lower triangular matrix
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)