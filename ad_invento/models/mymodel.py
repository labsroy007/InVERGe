
import torch
from torch import nn
import gc

from ijepa.src.models.vision_transformer import vit_giant, vit_base, vit_huge, vit_large, vit_small, vit_tiny
from ijepa.src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
import copy

'''
from transformers import BlipForConditionalGeneration
MODEL_NAME = "Salesforce/blip-image-captioning-large"

model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

pretrain_path = "Apply encoder checkpoint" # "encoder_checkpoint.pt"

checkpoint = torch.load(pretrain_path, map_location='cpu') 
  
print("Load pre-trained checkpoint from: %s" % pretrain_path) 
checkpoint_model = checkpoint
state_dict = model.state_dict() 
# for k in state_dict.keys():
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape: 
#         print(f"Removing key {k} from pretrained checkpoint") 
#         del checkpoint_model[k] 
# if False: 
#     for k in ['fc_norm.weight', 'fc_norm.bias']: 
#         try: 
#             del checkpoint_model[k] 
#         except: 
#             pass 


# interpolate position embedding 
# interpolate_pos_embed(model, checkpoint_model) 

# load pre-trained model 
msg = model.load_state_dict(checkpoint_model, strict=False) 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
model.text_decoder.cls=Identity()

'''

import torch
import sys
sys.path.append('ijepa_path')
import src.models


checkpoint_path = r"Apply encoder checkpoint" # "encoder_checkpoint.pt"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load the model architecture
my_model = checkpoint['model_architecture']

# Load the model's state_dict
my_model.load_state_dict(checkpoint['model_state_dict'])

# Load the optimizer's state_dict (if needed)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Access other saved information (e.g., epoch number, loss)
# epoch = checkpoint['epoch']

# Ensure the model is in evaluation mode (if necessary)
# mymodel.eval()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
my_model.text_decoder.cls=Identity()

import torch.cuda as cuda

gc.collect()
cuda.empty_cache()
cuda.synchronize()
