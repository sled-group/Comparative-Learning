import os
import json
import torch
import clip
import time
import pickle
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from PIL import Image


from config import *
from dataset import *
from my_models import *
from util import *

def get_training_data(in_path):
	path = os.path.join(in_path, 'train_new_objects_dataset.json')
	with open(path, 'r') as file:
		# Load JSON data from the file
		training_data = json.load(file)
	return training_data

def get_batches(base_names, in_path, source):
	images = []
	for base_name in base_names:
		path = os.path.join(in_path, source, base_name+'_rgba.pickle')
		with open(path, 'rb') as file:
			emb = pickle.load(file)
			images.append(emb)
	return images

def my_train_clip_encoder(training_data, memory, in_path, out_path, source, model_name):
	# get model
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.train().to(device)
	centroid_sim = torch.rand(1, latent_dim).to(device)

	loss_sim = None
	loss_dif = None
	loss = 10

	t_tot = 0
	t_start = time.time()
	previous_lesson = None

	for batch in training_data:
		attr = batch['attribute']
		lesson = batch['lesson']
		
		if lesson != previous_lesson and previous_lesson != None:
			############ print loss ############
			print(loss.detach().item(), loss_sim.detach().item(),loss_dif.detach().item())
			############ save model ############
			with torch.no_grad():
				memory[previous_lesson] = {'model': model.to('cpu').state_dict(),
								'arch': ['Filter', ['para_block1']],
								'centroid': centroid_sim.to('cpu')
								}
			#with open(os.path.join(out_path, model_name+'_'+str(n_split)+'.pickle'), 'wb') as handle:
			with open(os.path.join(out_path, model_name+'.pickle'), 'wb') as handle:
				pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
			############ print time ############
			t_end = time.time()
			t_dur = t_end - t_start
			t_tot += t_dur
			print("Time: ", t_dur, t_tot)
			############ reset model ############
			if lesson in memory.keys():
				print("______________ loading_____________________")
				model.load_state_dict(memory[lesson]['model'])
				optimizer = optim.Adam(model.parameters(), lr=lr)
				model.train().to(device)
				centroid_sim = torch.rand(1, latent_dim).to(device)
			else:
				clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
				model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
				optimizer = optim.Adam(model.parameters(), lr=lr)
				model.train().to(device)
				centroid_sim = torch.rand(1, latent_dim).to(device)
		if lesson != previous_lesson:
			print("#################### Learning: " + str(lesson))
			
		previous_lesson = lesson

		base_names_sim = batch['base_names_sim']
		base_names_dif = batch['base_names_dif']

		# Get Inputs: sim_batch, (sim_batch, 4, 132, 132)
		images_sim = get_batches(base_names_sim, in_path, source)
		images_sim = torch.stack(images_sim, dim=0)
		images_sim = images_sim.to(device)

		# run similar model
		z_sim = model(images_sim)
		centroid_sim = centroid_sim.detach()
		centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))

		# Run Difference
		images_dif = get_batches(base_names_dif, in_path, source)
		images_dif = torch.stack(images_dif, dim=0)
		images_dif = images_dif.to(device)

		# run difference model
		z_dif = model(images_dif)
		loss_dif = get_sim_not_loss(centroid_sim, z_dif)

		# compute loss
		loss = (loss_sim)**2 + (loss_dif-1)**2

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return memory

def my_clip_train(in_path, out_path, n_split, model_name, source):  	
	# load training data
	training_data = get_training_data(in_path)
	# load encoder models from memory
	memory = {}
	
	if n_split == '0':
		learning_list = types_logical_with_learning
	elif n_split == '1':
		learning_list = types_logical_with_learning_1
	elif n_split == '2':	
		learning_list = types_logical_with_learning_2
	elif n_split == '3':
		learning_list = types_logical_with_learning_3
	elif n_split == '4':
		learning_list = types_logical_with_learning_4
	elif n_split == '5':
		learning_list = types_logical_with_learning_5
	elif n_split == '6':
		learning_list = types_logical_with_learning_6
	elif n_split == '7':
		learning_list = types_logical_with_learning_7

	memory = my_train_clip_encoder(training_data, memory, in_path, out_path, source, model_name)


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--in_path', '-i',
				help='Data input path', required=True)
	
	argparser.add_argument('--out_path', '-o',
				help='Model memory output path', required=True)

	argparser.add_argument('--n_split', '-s', default=0,
				help='Split number', required=None)
	
	argparser.add_argument('--model_name', '-n', default='first_try_model',
				help='Best model memory to be saved file name', required=False)
	
	argparser.add_argument('--gpu_idx', '-g', default=0,
				help='Select gpu index', required=False)
	
	args = argparser.parse_args()
	device = "cuda" if torch.cuda.is_available() else "cpu"	
	#gpu_index = int(args.gpu_idx)
	#torch.cuda.set_device(gpu_index)
	#print('gpu:',gpu_index)
		
	my_clip_train(args.in_path, args.out_path, args.n_split, args.model_name, 'train/')
