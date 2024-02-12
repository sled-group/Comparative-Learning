#%%
import os
import clip 
from PIL import Image
import torch 

from config import *
from util import *
from my_models import *

clip_model, clip_preprocessor = clip.load("ViT-B/32")
clip_model.eval()

   
file_path1 = '/Users/filippomerlo/Desktop/Datasets/SOLA/novel_train/aqua_glass_cone_shade_base_stretch_normal_scale_large_brightness_bright_view_-2_-2_2_depth.png'
file_path2 = '/Users/filippomerlo/Desktop/Datasets/SOLA/novel_train/aqua_glass_cone_shade_base_stretch_normal_scale_large_brightness_bright_view_-2_-2_2_rgba.png'
        
image1 = Image.open(file_path1)
image2 = Image.open(file_path2)
image1 = clip_preprocessor(image1).unsqueeze(0)
image2 = clip_preprocessor(image2).unsqueeze(0)

emb1 = clip_model.encode_image(image1)
emb2 = clip_model.encode_image(image2)

centroid_sim = torch.rand(1, latent_dim).to(device)
#%%
emb = [emb1, emb2]
emb = torch.stack(emb, dim = 0 )
print(emb.shape)
model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
model.eval()
z_sim = model(emb)
print(z_sim.shape)
centroid_sim = centroid_sim.unsqueeze(dim=0).detach()
print(centroid_sim.shape)
centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))
print(centroid_sim.shape)