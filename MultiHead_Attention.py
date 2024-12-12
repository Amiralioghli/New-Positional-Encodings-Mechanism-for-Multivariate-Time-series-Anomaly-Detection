import math
import torch
from torch import nn
import torch.nn.functional as F
 
    
def scaled_dot_product_attention(q , k , v, mask = None):
    d_k = torch.tensor(q.shape[-1])
    scaled = torch.matmul(q , k.transpose(-1 , -2)) / math.sqrt(d_k)

    if mask is not None:
        scaled =scaled + mask
    attention = F.softmax(scaled , dim = -1)
    values = torch.matmul(attention , v)

    return values , attention

class MultiheadAttention(nn.Module):
    def __init__(self, d_model , num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = int(d_model // num_heads)

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.qkv_linear = nn.Linear(d_model, d_model)

    def forward(self , x , mask = None):

        batch_size , sequence_length , input_size = x.size()

        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size , sequence_length , self.num_heads , 3 * self.head_dim)
        qkv = qkv.permute(0 , 2 , 1 , 3)
        q , k , v = qkv.chunk(3 , dim = -1)
        values , attention = scaled_dot_product_attention(q, k, v , mask)
        values = values.reshape(batch_size , sequence_length , self.num_heads * self.head_dim)
        out = self.qkv_linear(values)
        return out
    
    
    
    
    
    