import torch.nn as nn
from typing import Dict

class LoRABaseLayer:

    def __init__(self,rank:int,lora_alpha:int,lora_p:float,use_rslora:bool) -> None:

        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_drop = nn.Dropout(lora_p)
        self.use_rslora = use_rslora

        self.scaling = lora_alpha / rank ** 0.5 if use_rslora else lora_alpha / rank

    def _load_pretrained_weights(self,state_dict:Dict) -> None:

        self.weight.data = state_dict['weight']

        if 'bias' in state_dict.keys():

            self.bias.data = state_dict['data'] 