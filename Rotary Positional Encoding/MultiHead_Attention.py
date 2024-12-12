import math
import torch
from torch import nn
import torch.nn.functional as F
from Feed_Forward_and_Positional_Encoding import RotaryPositionalEmbedding

# def get_rotary_matrix(context_len, embedding_dim):
  
#     R = torch.zeros((context_len, embedding_dim, embedding_dim), requires_grad=False)
#     positions = torch.arange(1, context_len+1).unsqueeze(1)
#     # Create matrix theta (shape: context_len  x embedding_dim // 2)
#     slice_i = torch.arange(0, embedding_dim // 2)
#     theta = 10000. ** (-2.0 * (slice_i.float()) / embedding_dim) 
#     m_theta = positions * theta
#     # Create sin and cos values
#     cos_values = torch.cos(m_theta)
#     sin_values = torch.sin(m_theta)
#     # Populate the rotary matrix R using 2D slicing
#     R[:, 2*slice_i, 2*slice_i] = cos_values
#     R[:, 2*slice_i, 2*slice_i+1] = -sin_values
#     R[:, 2*slice_i+1, 2*slice_i] = sin_values
#     R[:, 2*slice_i+1, 2*slice_i+1] = cos_values
#     return R.to(device=torch.device('mps'))


    
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

        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        
        self.qkv_linear = nn.Linear(d_model, d_model)
        
        self.rotary_position = RotaryPositionalEmbedding(d_model = d_model, max_seq_len=64)

    def forward(self , x , mask = None):

        batch_size , sequence_length , input_size = x.size()
        
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        
        #R_matrix = get_rotary_matrix(context_len = 187, embedding_dim=200)
        q = self.rotary_position(q)
        k = self.rotary_position(k)
        #v = self.rotary_position(v)
        
        q = q.reshape(batch_size , sequence_length , self.num_heads , self.head_dim)
        k = k.reshape(batch_size , sequence_length , self.num_heads , self.head_dim)
        v = v.reshape(batch_size , sequence_length , self.num_heads , self.head_dim)
        q = q.permute(0 , 2 , 1 , 3)
        k = k.permute(0 , 2 , 1 , 3)
        v = v.permute(0 , 2 , 1 , 3)
        
        values , attention = scaled_dot_product_attention(q, k, v , mask)
        values = values.reshape(batch_size , sequence_length , self.num_heads * self.head_dim)
        out = self.qkv_linear(values)
        return out
    
    
    
    
    
    