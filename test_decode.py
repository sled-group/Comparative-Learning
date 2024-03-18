#%%
import pickle
import torch 
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import clip 
from models import Decoder
from config import *
from dataset import MyDataset
from util import *
from pprint import pprint
import random 

random.seed(42)

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated
device = torch.device('mps')

memory_path = '/Users/filippomerlo/Desktop/memories/mem_decoder_logic_small_mse.pickle'

with open(memory_path, 'rb') as f:
        memory_base = pickle.load(f)

def get_key_from_value(dictionary, target_value):
    target = ''
    for key, value in dictionary.items():
        for v in value:
            if v == target_value:
                target = key
    return target 

in_path = '/Users/filippomerlo/Desktop/Datasets/SOLA'
source = 'train'
in_base = bn_train
types = ['rgba']
dic = dic_train_logical
vocab = vocabs

clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
clip_model.eval()

dt = MyDataset(in_path, source, in_base, types, dic, vocab, clip_preprocessor)
#for k,v, in rules.items():
#    print(v)
#    a,b,c,d = dt.get_batches_for_rules(v, batch_size = 1, force_rule = True)
#    for i, _ in enumerate(a):
#        print(a[i])
#        print(c[i])

data_loader = DataLoader(dt, batch_size=100, shuffle=True)

train_labels, train_features = next(iter(data_loader))
_, idxs = train_labels.topk(3)
idxs, _ = torch.sort(idxs)

# some operations
with torch.no_grad():
    acc = dict()
    n_trials_per_attr = dict()
    n_trials = 100
    for trial in range(n_trials):
        # get samples for the trial 
        # get their one-hot encoded features
        train_labels, train_features = next(iter(data_loader))
        _, idxs = train_labels.topk(3)
        idxs, _ = torch.sort(idxs)
        # encode the images with clip
        ans = []
        for i,im in enumerate(train_features):
            ans.append(clip_model.encode_image(im.unsqueeze(0).to(device)).squeeze(0))
        ans = torch.stack(ans)

        # get the answers
        #for attr in types_learning:
        for lesson in memory_base.keys():
            if 'decoder' in memory_base[lesson].keys(): 
                # check if in the image bacth there is an image that has the attr
                # if not, skip the attr
                split_lesson = lesson.split()   
                rels = ['and', 'or', 'not']
                attrs = [x for x in split_lesson if x not in rels]
                attrs_coded = [vocabs.index(x) for x in attrs]
                check = True
                for obj in idxs:
                    tresh = len(attrs_coded)
                    c = 0
                    for attr in attrs_coded:
                        if attr in obj:
                            c += 1
                    if c == tresh:
                        check = False
                        break
                if check:
                    continue
                # get attr categoy
                attr = get_key_from_value(dic, lesson)
                # fill acc dict
                if attr not in acc.keys():
                    acc[attr] = 0
                    n_trials_per_attr[attr] = 0
                n_trials_per_attr[attr] += 1
                answers = dict()
                #for lesson in dic[attr]:
                centroid = memory_base[lesson]['centroid'].to(device)
                dec = Decoder(latent_dim).to(device)
                dec.load_state_dict(memory_base[lesson]['decoder'])
                decoded_rep = dec(centroid)
                C = decoded_rep.repeat(ans.shape[0], 1)
                disi = ((ans - C)**2).mean(dim=1).detach().to('cpu')
                v, topk_idxs = disi.topk(1, largest=False)
                answers[lesson] = [idxs[i] for i in topk_idxs]

                for lesson in answers.keys():
                    for coded in answers[lesson]:
                        color = vocabs[coded[0]]
                        material = vocabs[coded[1]]
                        shape = vocabs[coded[2]]
                        if 'and' in lesson.split():
                            l1 = lesson.split()[0]
                            l2 = lesson.split()[2]
                            if l1 in [color, material, shape] and l2 in [color, material, shape]:
                                acc[attr] += 1
                        elif 'or' in lesson.split():
                            l1 = lesson.split()[0]
                            l2 = lesson.split()[2]
                            if l1 in [color, material, shape] or l2 in [color, material, shape]:
                                acc[attr] += 1
                        elif 'not' in lesson.split():
                            l1 = lesson.split()[1]
                            if l1 not in [color, material, shape]:
                                acc[attr] += 1
                        else:
                            if lesson in [color, material, shape]:
                                acc[attr] += 1

# print the results
import matplotlib.pyplot as plt

# Accuracy values
categories = []
accuracies = []

for k in acc.keys():
    print(f'{k}: ',acc[k]/n_trials_per_attr[k])
    categories.append(k)
    accuracies.append(acc[k]/n_trials_per_attr[k])

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(categories, accuracies)
plt.ylim(0, 1)  # Setting y-axis limits to represent accuracy values between 0 and 1
plt.title('Accuracy Metrics')
plt.xlabel('Categories')
plt.ylabel('Accuracy')
plt.show()

#%% TRY TO DO ALGEBRIAC OPERATIONS WITH LOGICAL 

for k in memory_base.keys():
    if 'decoder' in memory_base[k].keys():
        print(k)

#%%
data_loader = DataLoader(dt, batch_size=200, shuffle=True)
train_labels, train_features = next(iter(data_loader))
_, idxs = train_labels.topk(3)
idxs, _ = torch.sort(idxs)
ans = []
for i,im in enumerate(train_features):
    ans.append(clip_model.encode_image(im.unsqueeze(0).to(device)).squeeze(0))
ans = torch.stack(ans)
#%%
query1 = 'white or cube'
query2 = 'not white'
query3 = 'not red'
query4 = 'rubber and cube'
complete_query = [query4]

reps = []
for q in complete_query:
    # get reps decoded
    centroid = memory_base[q]['centroid'].to(device)
    dec = Decoder(latent_dim).to(device)
    dec.load_state_dict(memory_base[q]['decoder'])
    decoded_rep = dec(centroid)
    reps.append(decoded_rep)

answers = dict()
query_rep = torch.stack(reps).sum(dim=0)
C = query_rep.repeat(ans.shape[0], 1)
disi = ((ans - C)**2).mean(dim=1).detach().to('cpu')
v, topk_idxs = disi.topk(10, largest=False)
answers[q] = [idxs[i] for i in topk_idxs]
for lesson in answers.keys():
    for coded in answers[lesson]:
        color = vocabs[coded[0]]
        material = vocabs[coded[1]]
        shape = vocabs[coded[2]]
        print(color, material, shape)
