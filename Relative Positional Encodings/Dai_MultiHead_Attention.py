import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ========= Resource  
#     "Dai, Z., Yang, Z., Yang, Y., Carbonell, J. G., Le, Q. V., & Salakhutdinov, R. (2019). 
#     Transformer-XL: Attentive language models beyond a fixed-length context. CoRR, abs/1901.02860. [Online]. A
#     vailable: http://arxiv.org/abs/1901.02860, Transformer-XL
# ==========

class RelativeMultiHeadAttention_2019_Dai(nn.Module):
  
    def __init__(
            self,
            d_model: int = 200,
            num_heads: int = 2,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention_2019_Dai, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)    
        self.value_proj = nn.Linear(d_model, d_model)  
        self.pos_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout_p)
        self.out_proj = nn.Linear(d_model, d_model)  

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask=None,
    ) -> torch.Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score
