
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self,d_model, seq_len , details, n_classes: int = 2):
      super().__init__()
      self.norm = nn.LayerNorm(d_model)
      self.details = details
      self.flatten = nn.Flatten()
      #self.Linear1 = nn.Linear(d_model * seq_len, 512)
      #self.relu1 = nn.ReLU()
      #self.Linear2 = nn.Linear(512, 256)
      #self.relu2 = nn.ReLU()
      #self.Linear3 = nn.Linear(256, 128)
      #self.relu3 = nn.ReLU()
      self.out = nn.Linear(d_model * seq_len, n_classes)
    
      
    def forward(self,x):

      if self.details:  print('in classification head : '+ str(x.size())) 
      x = self.norm(x)
      x = self.flatten(x)
      #x = self.Linear1(x)
      #x = self.relu1(x)
      #x = self.Linear2(x)
      #x = self.relu2(x)
      #x = self.Linear3(x)
      #x = self.relu3(x)
      out = self.out(x)
      return out
  