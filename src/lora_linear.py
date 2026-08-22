import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_layer import LoRABaseLayer

class LoRALinearLayer(nn.Linear,LoRABaseLayer):

    def __init__(self,in_features:int,out_features:int,bias:bool,rank:int,lora_alpha:int,lora_p:float,use_rslora:bool,**kwargs) -> None:

        nn.Linear.__init__(self,in_features,out_features,bias=bias,**kwargs)
        LoRABaseLayer.__init__(self,rank,lora_alpha,lora_p,use_rslora)

        assert rank > 0, "Rank must be greater than zero"

        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(in_features,rank))
        self.lora_B = nn.Parameter(torch.zeros(rank,out_features))

        self._initialize_weights(self.lora_A)

    def _initialize_weights(self,layer:nn.Parameter) -> None:

        nn.init.kaiming_uniform_(layer,a = math.sqrt(5))

    def _merge_weights(self) -> nn.Linear:

        merged_weight = self.weight.data + (self.lora_A @ self.lora_B).T * self.scaling
        state_dict = {"weight" : merged_weight}

        if self.bias is not None:

            state_dict["bias"] = self.bias.data

        merged_linear = nn.Linear(self.in_features,self.out_features,bias = True if self.bias is not None else False)
        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        original_out = F.linear(x,self.weight,bias = self.bias)
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_drop(x) @ lora_mult  
        output = original_out + low_rank_out

        return output        
   