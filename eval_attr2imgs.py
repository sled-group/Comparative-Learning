#%%
from config import *
from dataset import *
from models import *
from tqdm import tqdm

from torch.utils.data import DataLoader
import pickle
import clip
import argparse

from pprint import pprint


def my_clip_evaluation(in_path, source, memory_path, in_base, types, dic, vocab_data, vocab_eval):

	with torch.no_grad():
		with open(memory_path, 'rb') as f:
			memory = pickle.load(f)

		# get dataset
		clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
		dt = MyDataset(in_path, source, in_base, types, dic, vocab_data,
					clip_preprocessor=clip_preprocess)
		data_loader = DataLoader(dt, batch_size=132, shuffle=True)

		top3 = 0
		top3_color = 0
		tot_color = 0
		top3_material = 0
		tot_material = 0
		top3_shape = 0
		tot_shape = 0
		top3_and = 0
		tot_and = 0
		top3_or = 0
		tot_or = 0
		top3_not = 0
		tot_not = 0

		top3_logic = 0
		tot_num = 0
		tot_num_logic = 0

		#for base_is, images in data_loader: # labels (one hot), images (clip embs)
		base_is, images = next(iter(data_loader))
		
		# Prepare the inputs
		images = images.to(device)
		ans = []
		batch_size_i = len(base_is)

		# go through memory
		for label in tqdm(vocab_eval, desc="Processing labels", unit="label"): # select a label es 'red'
			if label not in memory.keys():
				print('nope')
				ans.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
				continue

			# load model
			model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
			model.load_state_dict(memory[label]['model']) # load weights corresponding to red
			model.to(device)
			model.eval() # freeze

			# load centroid
			centroid_i = memory[label]['centroid'].to(device)
			centroid_i = centroid_i.repeat(batch_size_i, 1)

			# compute stats
			z = model(clip_model, images).squeeze(0)
			disi = ((z - centroid_i)**2).mean(dim=1) # distance between z and centroid_i
			ans.append(disi.detach().to('cpu'))

			#####

			dsi = disi.detach().to('cpu')
			values, indices = dsi.topk(1, largest=False) # get top indices
			_, indices_lb = base_is.topk(3)
			indices_lb, _ = torch.sort(indices_lb)

			for bi in indices:
				color = indices_lb[bi][0]
				material = indices_lb[bi][1]
				shape = indices_lb[bi][2]
				attrs = [color, material, shape]

				if len(label.split())<2:
					tot_num += 1
					l1 = label
					l1_idx = vocab_data.index(l1)
					if color == l1_idx:
						top3_color += 1
					if material == l1_idx:
						top3_material += 1
					if shape == l1_idx:
						top3_shape += 1
					
				else:
					tot_num_logic += 1
					if 'and' in label.split():
						tot_and += 1
						l1 = label.split()[0]
						l2 = label.split()[2]
						l1_idx = vocab_data.index(l1)
						l2_idx = vocab_data.index(l2)
						if (l1_idx in attrs) and (l2_idx in attrs):
							top3_and += 1
					elif 'or' in label.split():
						tot_or += 1
						l1 = label.split()[0]
						l2 = label.split()[2]
						l1_idx = vocab_data.index(l1)
						l2_idx = vocab_data.index(l2)
						if (l1_idx in attrs) or (l2_idx in attrs):
							top3_or += 1
					elif 'not' in label.split():
						tot_not += 1
						l1 = label.split()[1]
						l1_idx = vocab_data.index(l1)
						if l1_idx not in attrs:
							top3_not += 1
		top3 = (top3_color + top3_material + top3_shape)/tot_num
		top3_logic = (top3_and + top3_or + top3_not)/tot_num_logic
		print('color',top3_color,'\nmaterial',top3_material,'\nshape',top3_shape,'\nclassic',top3)
		print('and',top3_and/tot_and,'\nor',top3_or/tot_or,'\nnot',top3_not/tot_not,'\nlogic',top3_logic)
			#####

source = 'train'
in_base = bn_train
types = ['rgba']
dic = dic_train_logical
vocab_eval = all_vocabs
vocab_data = vocabs

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()

	argparser.add_argument('--in_path', '-i',
				help='Data input path', required=True)

	argparser.add_argument('--memory_path', '-m',
				help='Memory input path', required=True)

	args = argparser.parse_args()
	
	#top3, top3_and, top3_or, top3_not = my_clip_evaluation(args.in_path, source, args.memory_path, in_base, types, dic, vocab_data, vocab_eval)
	my_clip_evaluation(args.in_path, source, args.memory_path, in_base, types, dic, vocab_data, vocab_eval)
	#print('top3:', top3)
	#print('top3_and:', top3_and)
	#print('top3_or:', top3_or)
	#print('top3_not:', top3_not)


# load model
#clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
#dt = MyDataset(in_path, source, in_base, types, dic, vocab_data,
#					clip_preprocessor=clip_preprocess)
#data_loader = DataLoader(dt, batch_size=10, shuffle=True)
#base_is, images = next(iter(data_loader))
#
#labels = ['red', 'green', 'blue', 'aqua']
#batch_size_i = 10
#ans = []
#for label in labels:
#	model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
#	model.load_state_dict(memory[label]['model']) # load weights corresponding to red
#	model.to(device)
#	model.eval() # freeze
#
#	# load centroid
#	centroid_i = memory[label]['centroid'].to(device)
#	centroid_i = centroid_i.repeat(batch_size_i, 1)
#
#	# compute stats
#	z = model(clip_model, images).squeeze(0)
#	disi = ((z - centroid_i)**2).mean(dim=1)
#	ans.append(disi.detach().to('cpu'))

#ans = torch.stack(ans, dim=1)
#pprint(ans)
#values, indices = ans.topk(3, largest=False)
#pprint(indices)
#_, indices_lb = base_is.topk(3)
#indices_lb, _ = torch.sort(indices_lb)
#pprint(indices_lb)