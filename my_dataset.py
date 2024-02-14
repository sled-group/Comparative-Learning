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
		self.dic_without_logical = {k:v for k,v in self.dic.items() if ' ' not in k}
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

		# Add a filter for names not allowed in the train set:
		# define a set of objects to remove
		
		self.name_set_filtered = self.name_set.copy()
		for name in self.name_set:
			split = name.split('_')
			c = split[0]
			m = split[1]
			s = split[2]
			if (c,m,s) in remove_from_train:
				self.name_set_filtered.remove(name)

	def __len__(self):
		return len(self.names_list)

	# only for CLIP emb
	def __getitem__(self, idx):
		base_name = self.names_list[idx]

		# get label indicies
		nm = pareFileNames(base_name)
		num_labels = [self.vocab_nums[li] for li in [nm['color'],
						nm['material'], nm['shape']]]

		#  turn num_labels into one-hot
		labels = torch.zeros(len(self.vocab))
		for xi in num_labels:
			labels[xi] = 1

		return labels, base_name

	