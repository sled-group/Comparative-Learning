#%%
import os
import json
import clip 
from config import *
from dataset import MyDataset
from util import *
import argparse
from tqdm import tqdm
from pprint import pprint

# Function to load a list from a file using json
def load_list(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        # If the file does not exist yet, return an empty list
        return []

# Function to save a list to a file using json
def save_list(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        for v in value:
            if v == target_value:
                return key
    return None 

# Build the dataset object
def get_datasets(in_path,out_path):
    parameters_list = [
        ['train', bn_train, ['rgba'], dic_train_logical, True, 'train_new_objects'],
        ['train', bn_train, ['rgba'], dic_train_logical, False, 'train_var'],
    ]

    vocab = vocabs

    for parameters in parameters_list:
        source, in_base, types, dic, train, dataset_name = parameters

        new_out_path = os.path.join(out_path, dataset_name+'_dataset.json')
        save_list(new_out_path, []) ## After doing this one time, comment this line
        dt = MyDataset(in_path, source, in_base, types, dic, vocab)
        new_batches = []

        for lesson in tqdm(all_vocabs):
            attribute = get_key_from_value(dic_train_logical, lesson)
            
            for i in range(500):               
                base_names_sim, base_names_dif = dt.get_paired_batches_names(attribute, lesson, 132, train)
                new_batches.append(
                    {
                    'attribute' : attribute,
                    'lesson' : lesson,
                    'base_names_sim' : base_names_sim,
                    'base_names_dif' : base_names_dif
                    }
                )
        all_lessons = load_list(new_out_path)
        all_lessons += new_batches
        save_list(new_out_path, all_lessons)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get datasets')
    parser.add_argument('--in_path', type=str, help='Path to the dataset')
    parser.add_argument('--out_path', type=str, help='Path to the output')
    args = parser.parse_args()
    
    get_datasets(args.in_path,args.out_path)
