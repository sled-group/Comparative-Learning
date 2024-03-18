#%%
from pprint import pprint
from config import *
file_path = '/Users/filippomerlo/Desktop/Datasets/SOLA/names/bn_train.txt'
with open(file_path, 'r') as file:
        names = file.read().split('\n')
names.remove('')
pprint(names)
#%%
train_names = []
test_names = []
for name in names:
    split = name.split('_')
    c = split[0]
    m = split[1]
    s = split[2]
    if (c,m,s) in remove_from_train:
        test_names.append(name)
    else:
        train_names.append(name)
          
print(len(test_names))
print(len(train_names))

#%%
train_names_path = '/Users/filippomerlo/Desktop/Datasets/SOLA/names/new_obj_train.txt'
test_names_path = '/Users/filippomerlo/Desktop/Datasets/SOLA/names/new_obj_test.txt'

with open(train_names_path, 'w') as file:
    # Write each item in the list as a separate line
    for item in train_names:
        file.write(f"{item}\n")

with open(test_names_path, 'w') as file:
    # Write each item in the list as a separate line
    for item in test_names:
        file.write(f"{item}\n")