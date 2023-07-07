import os
import torch
import random
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT
from torch.utils.data.dataset import Dataset

from config import *
from util import *


class MyDataset():
	def __init__(self, in_path, source, in_base, types,
					dic, vocab, clip_preprocessor=None):
		self.dic = dic
		self.source = source
		self.types = types
		self.in_path = in_path
		self.totensor = TT.ToTensor()
		self.resize = TT.Resize((resize, resize))
		self.clip_preprocessor = clip_preprocessor

		# convert vocab list to dic
		self.vocab = vocab
		self.vocab_nums = {xi: idx for idx, xi in enumerate(self.vocab)}

		# Get list of test images
		self.names_list = []
		with open(os.path.join(self.in_path, 'names', in_base)) as f:
			lines = f.readlines()
			for line in lines:
				self.names_list.append(line[:-1])

		self.name_set = set(self.names_list)

	def __len__(self):
		return len(self.names_list)

	# only for CLIP emb
	def __getitem__(self, idx):
		base_name = self.names_list[idx]
		image = self.img_emb(base_name)

		# get label indicies
		nm = pareFileNames(base_name)
		num_labels = [self.vocab_nums[li] for li in [nm['color'],
						nm['material'], nm['shape']]]

		#  turn num_labels into one-hot
		labels = torch.zeros(len(self.vocab))
		for xi in num_labels:
			labels[xi] = 1

		return labels, image

	def img_emb(self, base_name):
		# get names
		names = []
		for tp in self.types:
			names.append(os.path.join(self.in_path, self.source,
							base_name + '_' + tp + '.png'))

		# if clip preprocess
		if self.clip_preprocessor is not None:
			images = self.clip_preprocessor(Image.open(names[0]))
			return images

		# preprocess images
		images = []
		for ni in range(len(names)):
			input_image = Image.open(names[ni]).convert('RGB')
			input_image = self.totensor(input_image)

			if names[ni][-16:] == "segmentation.png":
				input_image = input_image.sum(dim=0)
				vals_seg = torch.unique(input_image)
				seg_map = []

				# generate one hot segmentation mask
				for i in range(len(vals_seg)):
					mask = input_image.eq(vals_seg[i])
					# hack: only keep the non-background segmentation masks
					if mask[0][0] is True:
						continue
					seg_mapi = torch.zeros([input_image.shape[0],
						input_image.shape[1]]).masked_fill_(mask, 1)
					seg_map.append(seg_mapi)

				seg_map = torch.cat(seg_map).unsqueeze(0)
				images.append(seg_map)
			else:
				images.append(input_image)

			images[ni] = self.resize(images[ni])

		# (d, resize, resize), d = 3 + #objs (+ other img types *3)
		images = torch.cat(images)
		return images

	def get_better_similar(self, attribute, lesson):
		base_names = []
		images = []
		while len(base_names) < sim_batch:
			names_dic = {}
			for k, v in self.dic.items():
				if k == attribute:
					names_dic[k] = lesson
				else:
					names_dic[k] = random.choice(v)
			base_name = f'{names_dic["color"]}_{names_dic["material"]}_{names_dic["shape"]}_shade_{names_dic["shade"]}_stretch_{names_dic["stretch"]}_scale_{names_dic["scale"]}_brightness_{names_dic["brightness"]}_view_{names_dic["view"]}'

			if base_name in self.name_set:
				base_names.append(base_name)
				image = self.img_emb(base_name)
				images.append(image)

		images = torch.stack(images)
		return base_names, images

	def get_better_similar_not(self, attribute, lesson):
		base_names = []
		images = []
		while len(base_names) < sim_batch:
			names_dic = {}
			for k, v in self.dic.items():
				if k == attribute:
					tp = random.choice(v)
					while (tp == lesson):
						tp = random.choice(v)
					names_dic[k] = tp
				else:
					# all other attributes same
					tpo = random.choice(v)
					names_dic[k] = tpo
			base_name = f'{names_dic["color"]}_{names_dic["material"]}_{names_dic["shape"]}_shade_{names_dic["shade"]}_stretch_{names_dic["stretch"]}_scale_{names_dic["scale"]}_brightness_{names_dic["brightness"]}_view_{names_dic["view"]}'

			if base_name in self.name_set:
				base_names.append(base_name)
				image = self.img_emb(base_name)
				images.append(image)

		images = torch.stack(images)
		return base_names, images
