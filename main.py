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
	images = torch.stack(images, dim = 0)
	return images

def my_train_clip_encoder(training_data, n_split, memory, in_path, out_path, source, model_name):
	# Initialize model
	def initialize_model(lesson, memory):
		if lesson in memory.keys():
			print("______________ loading_____________________")
			model.load_state_dict(memory[lesson]['model'])
			optimizer = optim.Adam(model.parameters(), lr=lr)
			model.train().to(device)
			centroid_sim = torch.rand(1, latent_dim).to(device)
		else:
			model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
			optimizer = optim.Adam(model.parameters(), lr=lr)
			model.train().to(device)
			centroid_sim = torch.rand(1, latent_dim).to(device)
		print("#################### Learning: " + lesson)
		return model, optimizer, centroid_sim
	
	def save_model(model, previous_lesson, memory, n_split, t_tot):
		############ print loss ############
		print('loss:',loss.detach().item(), 'sim_loss:',loss_sim.detach().item(),'dif_loss:',loss_dif.detach().item())
		
		############ save model ############
		with torch.no_grad():
			memory[previous_lesson] = {'model': model.to('cpu').state_dict(),
							'arch': ['Filter', ['para_block1']],
							'centroid': centroid_sim.to('cpu')
							}
		with open(os.path.join(out_path, model_name+'_'+str(n_split)+'.pickle'), 'wb') as handle:
			pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		############ print time ############
		t_end = time.time()
		t_dur = t_end - t_start
		t_tot += t_dur
		print("Time: ", t_dur, t_tot)

		return memory, t_tot

	loss_sim = None
	loss_dif = None
	loss = 10

	t_tot = 0
	t_start = time.time()
	count = 0
	lesson = None
	previous_lesson = 'first_lesson'
	
	for i, batch in enumerate(training_data):
		
		# Get Lesson
		lesson = batch['lesson']

		# Init model if first lesson
		if previous_lesson == 'first_lesson':
			model, optimizer, centroid_sim = initialize_model(lesson,memory)

		# If we finished a lesson save it and initialize new model
		if lesson != previous_lesson and previous_lesson != 'first_lesson':
			memory, t_tot = save_model(model, previous_lesson, memory, n_split, t_tot)
			model, optimizer, centroid_sim = initialize_model(lesson,memory)
			count = 0
		
		previous_lesson = lesson
		count += 1

		# If loss < 0.008 skip all the remaining batches of the lesson
		# but it has to have done at least 1000 iterations
		if loss < 0.008 and count >= 1000:
			while lesson == previous_lesson:
				continue

		base_names_sim = batch['base_names_sim']
		base_names_dif = batch['base_names_dif']

		# Get Inputs: sim_batch, (sim_batch, 4, 132, 132)
		images_sim = get_batches(base_names_sim, in_path, source)
		images_sim = images_sim.to(device)

		# run similar model
		z_sim = model(images_sim)
		centroid_sim = centroid_sim.unsqueeze(dim=0).detach()
		centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))

		# Run Difference
		images_dif = get_batches(base_names_dif, in_path, source)
		images_dif = images_dif.to(device)

		# run difference model
		z_dif = model(images_dif)
		loss_dif = get_sim_not_loss(centroid_sim, z_dif)

		# compute loss
		loss = (loss_sim)**2 + (loss_dif-1)**2

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('B:',i,'L:',loss)

	memory, t_tot = save_model(model, previous_lesson, memory, n_split, t_tot)

	return memory

def my_clip_train(in_path, out_path, n_split, model_name, source):  	
	# load training data
	training_data = get_training_data(in_path)

	# select training data
	filtered_data = []
	for batch in training_data:
		if batch['attribute'] in attrs_split[n_split]:
			filtered_data.append(batch)
	training_data = None

	# load encoder models from memory
	memory = {}
	memory = my_train_clip_encoder(filtered_data, n_split, memory, in_path, out_path, source, model_name)


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
	gpu_index = int(args.gpu_idx)
	torch.cuda.set_device(gpu_index)
	print('gpu:',gpu_index)
		
	my_clip_train(args.in_path, args.out_path, args.n_split, args.model_name, 'train/')
