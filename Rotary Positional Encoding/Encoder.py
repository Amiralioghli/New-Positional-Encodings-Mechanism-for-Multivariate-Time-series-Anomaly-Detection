import torch
from torch import nn
from Feed_Forward_and_Positional_Encoding import PositionwiseFeedForward #, RotaryPositionalEmbedding
from Layer_Norm import LayerNormalization
from MultiHead_Attention import MultiheadAttention
from Class_Head import ClassificationHead
from torchinfo import summary
import numpy as np

device = torch.device("mps")


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob,details):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=n_head)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.details = details
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(x)
        
        if self.details: print('in encoder layer : '+ str(x.size()))
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        if self.details: print('in encoder after norm layer : '+ str(x.size()))
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        if self.details: print('in encoder after ffn : '+ str(x.size()))
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob,details, device):
        super().__init__()

        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head
                                                  ,details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x ): 
        for layer in self.layers:
            x = layer(x ) 
        return x
    
class Transformer(nn.Module):

    def __init__(self,device, d_model, n_head, max_len, seq_len,
                 ffn_hidden, n_layers, drop_prob, details =False):
        super().__init__() 
        self.device = device
        self.details = details 
        self.encoder_input_layer = nn.Linear(   
            in_features=40, 
            out_features=d_model 
            )
   
        #self.pos_emb = RotaryPositionalEmbedding(d_model = d_model , max_seq_len = seq_len)
        self.encoder = Encoder(d_model=d_model,
                                n_head=n_head, 
                                ffn_hidden=ffn_hidden, 
                                drop_prob=drop_prob,
                                n_layers=n_layers,
                                details=details,
                                device=device)
        self.classHead = ClassificationHead(seq_len=seq_len,d_model=d_model,details=details,n_classes=2)

    def forward(self, x):
        if self.details: print('before input layer: '+ str(x.size()) )
        x= self.encoder_input_layer(x)
        if self.details: print('after input layer: '+ str(x.size()) )
        #x = self.pos_emb(x)
        if self.details: print('after pos_emb: '+ str(x.size()) )
        enc_src = self.encoder(x) 
        cls_res = self.classHead(enc_src)
        if self.details: print('after cls_res: '+ str(cls_res.size()) )
        return cls_res

 

# device = torch.device("mps")
# sequence_len=187 # sequence length of time series
# max_len=5000 # max time series sequence length 
# n_head = 4 # number of attention head
# n_layer = 2 # number of encoder layer
# drop_prob = 0.1
# d_model = 200 # number of dimension ( for positional embedding)
# ffn_hidden = 512 # size of hidden layer before classification 
# feature = 1 # for univariate time series (1d), it must be adjusted for 1. 
# model =  Transformer(  d_model=d_model, details=False, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, device=device)

# batch_size = 7

# summary(model, input_size=(batch_size,sequence_len,feature) , device=device)

# print(model)











    
