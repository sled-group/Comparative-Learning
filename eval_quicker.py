#%%
import torch
import clip

from torch.utils.data import DataLoader

from config import *
from my_dataset import *
from my_models import *

import pickle
import argparse

from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

def get_batches(base_names, preprocessed_images_path, source):
	images = []
	for base_name in base_names:
		path = os.path.join(preprocessed_images_path, source, base_name+'_rgba.pickle')
		with open(path, 'rb') as file:
			emb = pickle.load(file)
			images.append(emb)
	images = torch.stack(images, dim = 0)
	return images

def my_clip_evaluation_base(in_path, preprocessed_images_path, source, memory, in_base, types, dic, vocab):
    with torch.no_grad():

        # get dataset
        dt = MyDataset(in_path, source, in_base, types, dic, vocab)
        data_loader = DataLoader(dt, batch_size=129, shuffle=True)

        top3 = 0
        top3_color = 0
        top3_material = 0
        top3_shape = 0
        tot_num = 0

        for base_is, names in data_loader:
            # Prepare the inputs
            images = get_batches(names, preprocessed_images_path, source)
            images = images.to(device)
            batch_size_i = len(base_is)
        
            ans = []
            # go through memory
            for label in vocabs:
                if label not in memory.keys():
                    ans.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
                    continue

                # load model
                model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
                model.load_state_dict(memory[label]['model'])
                model.to(device)
                model.eval()

                # load centroid
                centroid_i = memory[label]['centroid'].to(device)
                centroid_i = centroid_i.repeat(batch_size_i, 1)

                # compute stats
                z = model(images).squeeze(0)
                z = model(images).squeeze(1)
                disi = ((z - centroid_i) ** 2).mean(dim=1)
                ans.append(disi.detach().to('cpu'))

            # get top3 incicies
            ans = torch.stack(ans, dim=1)
            values, indices = ans.topk(3, largest=False)
            _, indices_lb = base_is.topk(3)
            indices_lb, _ = torch.sort(indices_lb)

            # calculate stats
            tot_num += len(indices)
            for bi in range(len(indices)):
                ci = 0
                mi = 0
                si = 0
                if indices_lb[bi][0] in indices[bi]:
                    ci = 1
                if indices_lb[bi][1] in indices[bi]:
                    mi = 1
                if indices_lb[bi][2] in indices[bi]:
                    si = 1

                top3_color += ci
                top3_material += mi
                top3_shape += si

                if (ci == 1) and (mi == 1) and (si == 1):
                    top3 += 1

        print('BASIC: ','Num:',tot_num, 
        'Color:',top3_color / tot_num, 'Material:',top3_material / tot_num, 'Shape:',top3_shape / tot_num, 'Tot:',top3 / tot_num)

    return top3 / tot_num

def my_clip_evaluation_logical(in_path, preprocessed_images_path, source, memory, in_base, types, dic, vocab, nk=106):
    with torch.no_grad():

        # get dataset
        dt = MyDataset(in_path, source, in_base, types, dic, vocab)
        data_loader = DataLoader(dt, batch_size=129, shuffle=True)

        tot_num = 0
        score_and = 0
        tot_num_and = 0
        errors_and = dict()

        score_or = 0
        tot_num_or = 0

        score_not = 0
        tot_num_not = 0

        tot_num_logical = 0

        for base_is, names in data_loader:

            # Prepare the inputs
            images = get_batches(names, preprocessed_images_path, source)
            images = images.to(device)
            batch_size_i = len(base_is)

            ans_logical = []
            # for each lable
            for label in logical_vocabs:
                if label not in memory.keys():
                    ans_logical.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
                    continue 
                # load model
                model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
                model.load_state_dict(memory[label]['model'])
                model.to(device)
                model.eval()

                # load centroid
                centroid_i = memory[label]['centroid'].to(device)
                centroid_i = centroid_i.repeat(batch_size_i, 1)

                # compute stats
                # embed all the images
                z = model(images).squeeze(0)
                z = model(images).squeeze(1)
                # compute the distance between each image embedded with the concept model and the concept centroid
                disi = ((z - centroid_i) ** 2).mean(dim=1)
                ans_logical.append(disi.detach().to('cpu'))
                # for each concept we have a list of distances between the images and the concept centroid
            
            # get top3 incicies
            ans_logical = torch.stack(ans_logical, dim=1)
            # for each image get the nk smallest distances indexes, so we can check the logical relations
            values, indices = ans_logical.topk(nk, largest=False) # 106 is the number of logical relations true for each image

            _, indices_lb = base_is.topk(3)
            indices_lb, _ = torch.sort(indices_lb)

            tot_num += len(indices)
            # calculate stats
            for bi in range(len(indices_lb)):
                # for each image get the features
                color = vocabs[indices_lb[bi][0]]
                material = vocabs[indices_lb[bi][1]]
                shape = vocabs[indices_lb[bi][2]]
                atrs = [color, material, shape]

                # check logical rep retrieved
                # for the nk logical relations associated with each image check the validy wrt the image 
                for i in indices[bi]:
                    tot_num_logical += 1
                    # check validity
                    prop = logical_vocabs[i].split(' ')

                    if 'not' in prop:
                        attr1 = prop[1]
                        attr2 = None
                        tot_num_not += 1   

                        if attr1 not in atrs:
                            score_not += 1

                    elif 'and' in prop:
                        attr1 = prop[0]
                        attr2 = prop[2]
                        tot_num_and += 1

                        if attr1 in atrs and attr2 in atrs:
                            score_and += 1
                        else:
                            def get_attr(lesson):
                                for k,v in dic_test_logical.items():
                                    if lesson in v:
                                        return k 
                            attr1 = get_attr(attr1)
                            attr2 = get_attr(attr2)
                            if attr1+'_and_'+attr2 not in errors_and.keys():
                                errors_and[attr1+'_and_'+attr2] = 1
                            errors_and[attr1+'_and_'+attr2] += 1


                    elif 'or' in prop:
                        attr1 = prop[0]
                        attr2 = prop[2]
                        tot_num_or += 1

                        if attr1 in atrs or attr2 in atrs:
                            score_or += 1

            tot_score_logical = score_not + score_and + score_or
        
        print('LOGICAL: ','Num:',tot_num,'Tot:',tot_score_logical / tot_num_logical, 
        'Not:',score_not / tot_num_not, 'And:',score_and / tot_num_and, 'Or:',score_or / tot_num_or)
        #print('AND errors:')
        #pprint(errors_and)

    return tot_score_logical/tot_num_logical, score_not / tot_num_not, score_and / tot_num_and, score_or / tot_num_or

#TESTING

types = ['rgba']
vocab = all_vocabs

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in_path', type=str, required=True)
    argparser.add_argument('--memory_path', type=str, required=True)
    argparser.add_argument('--preprocessed_images_path', type=str, required=True)
    args = argparser.parse_args()

    with open(args.memory_path, 'rb') as f:
            memory_complete = pickle.load(f)
    for i in range(2, 8):
        pieces = args.memory_path.split('first_try_model')
        new_path = pieces[0] + f'first_try_model_{i}.pickle'
        with open(new_path, 'rb') as f:
            memory = pickle.load(f)
        for k in memory.keys():
            memory_complete[k] = memory[k]

    # simpe concepts
    #print('*************')
    #print('mare new obj')
    #mare_new_obj = my_clip_evaluation_base(args.in_path, args.preprocessed_images_path, 'novel_test/', memory_complete, bn_n_test, types, dic_train, vocab)
    #print('*************')
    #print('mare var')
    #mare_var = my_clip_evaluation_base(args.in_path, args.preprocessed_images_path, 'test/', memory_complete, bn_test, types, dic_test, vocab)
    
    # logical concepts
    log_new_obj = []
    log_var = []
    for nk in range(1, 67):
        print(nk)
        mare_logical_new_obj = my_clip_evaluation_logical(args.in_path, args.preprocessed_images_path, 'novel_test/', memory_complete, bn_n_test, types, dic_train_logical, vocab, nk)
        mare_logical_var = my_clip_evaluation_logical(args.in_path, args.preprocessed_images_path, 'test/', memory_complete, bn_test, types, dic_test_logical, vocab, nk)
        log_new_obj.append(mare_logical_new_obj)
        log_var.append(mare_logical_var)
    
    import matplotlib.pyplot as plt

    # Assuming you have two lists named list1 and list2
    # Each element in the lists is a list of 4 scores: tot, not, and, or

    list1 = log_new_obj  # Replace [...] with your actual data
    list2 = log_var  # Replace [...] with your actual data

    # Extracting scores from each list
    tot1, not1, and1, or1 = zip(*list1)
    tot2, not2, and2, or2 = zip(*list2)

    # Plotting the scores
    plt.figure(figsize=(10, 6))

    plt.plot(tot1, label='Tot New Objects')
    plt.plot(not1, label='Not New Objects', linestyle='dashed')
    plt.plot(and1, label='And New Objects', linestyle='dashed')
    plt.plot(or1, label='Or New Objects', linestyle='dashed')

    plt.plot(tot2, label='Tot Variation', linestyle='dashdot')
    plt.plot(not2, label='Not Variation', linestyle='dotted')
    plt.plot(and2, label='And Variation', linestyle='dotted')
    plt.plot(or2, label='Or Variation', linestyle='dotted')

    plt.xlabel('Top k Logical Relations')
    plt.ylabel('Accuracy Scores')
    plt.title('Logical Relation Recognition')
    plt.legend()
    plt.grid(True)
    plt.savefig(pieces[0]+'plt.png')
    plt.show()
