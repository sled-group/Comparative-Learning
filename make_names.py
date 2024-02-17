#%%
from config import *
import os
import json
from pprint import pprint
remove_from_train = remove_from_train


print(len(remove_from_train))
count_dic = {}
for o in remove_from_train:
    name = "_".join(o)
    count_dic[name] = 0


folder_path = "/Users/filippomerlo/Desktop/Datasets/SOLA/names/bn_train.txt"
final_names = []
# List all files in the folder
with open(folder_path, 'r') as file:
    files = file.read().split("\n")

for name in files:
    file = name.split("_")
    attrs = file[0:4]
    if 'shade' in attrs:
       attrs.remove('shade')
    if 'torus' in attrs and 'knot' in attrs:
        attrs.remove('torus')
        attrs.remove('knot')
        attrs.append('torus_knot')
    for o in remove_from_train:
        if tuple(attrs) == o:
            final_names.append(name)
            name = "_".join(o)
            count_dic[name] += 1
pprint(final_names)

## Open the file in write mode ('w')
#with open('/Users/filippomerlo/Desktop/Datasets/SOLA/names/'+'no_test.txt', 'w') as file:
#    # Write each string from the list to the file
#    for string in final_names:
#        file.write(f"{string}\n")
#