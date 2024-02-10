from models import Decoder
from config import *
from dataset import MyDataset
from util import *

import argparse
import pickle
import torch 
from torch import nn
import os
import clip 
import random as rn
rn.seed(42)
import wandb

clip_model, preprocess = clip.load("ViT-B/32", device=device)

dec_types_logical_with_learning = [
 'color and material',
 'color and shape',
 'color or material',
 'color or shape',
 'not color',
 'material and shape',
 'material or shape',
 'not material',
 'not shape']

dec_dic_train_logical = dic_train_logical.copy()
for k in dec_types_logical_with_learning:
    if len(dic_train_logical[k]) > 5:
        dec_dic_train_logical[k] = rn.sample(dic_train_logical[k], 5)

dec_types_logical_with_learning += types_learning

def train_decoder(in_path, out_path, model_name, memory_path, 
                  clip_model, clip_preprocessor,
                  source, in_base, types, dic, vocab, 
                  epochs=2, lr=0.001, batch_size=108, latent_dim=16):
    
    batch_size = batch_size
    with open(memory_path, 'rb') as f:
        memory = pickle.load(f)
    
    dt = MyDataset(in_path, source, in_base, types, dic, vocab, 
                   clip_preprocessor)
    
    clip_model.eval()
    
    for attr in dec_types_logical_with_learning:
        print('learning' + ' ' + attr)
        for lesson in dic[attr]:
            wandb.init(project='decode logical', name=attr+': '+lesson)  # Replace with your project name and run name
            print(lesson)
            # build decoder
            dec = Decoder(latent_dim).to(device)

            # optimizer
            optimizer = torch.optim.Adam(dec.parameters(), lr=lr)
            # train decoder
            for epoch in range(epochs):

                for round in range(0,100):
                    operators = ['and', 'or', 'not']
                    if not any(op in lesson.split() for op in operators):
                        rec_name_sim, rec_img_sim, _ , _ = dt.get_paired_batches(attr, lesson, batch_size)
                        edit_name_sim, edit_img_sim, edit_name_diff , edit_img_diff = dt.get_paired_batches(attr, lesson, batch_size)
                        loss_list = list()

                        for i in range(0,batch_size):
                            
                            with torch.no_grad():
                                # load sample
                                rec_name = rec_name_sim[i]
                                rec_img = clip_model.encode_image(rec_img_sim[i].unsqueeze(0).to(device))

                                edit1_name = edit_name_sim[i]
                                edit1_img = clip_model.encode_image(edit_img_sim[i].unsqueeze(0).to(device))

                                edit2_name = edit_name_diff[i]
                                edit2_img = clip_model.encode_image(edit_img_diff[i].unsqueeze(0).to(device))

                                # get second attribute for edit: lesson_filter
        
                                lesson_filter = edit2_name.split('_')[0:3]
                                if 'color' in attr:
                                    lesson_filter = lesson_filter[0]
                                elif 'material' in attr:
                                    lesson_filter = lesson_filter[1]
                                elif 'shape' in attr:
                                    lesson_filter = lesson_filter[2]

                                # load centroid
                                centroid_i = memory[lesson]['centroid'].float().to(device)

                                # load filters
                                filter_i_edit = memory[lesson_filter]['model']['filter'].to(device)

                                filter_i_rec = memory[lesson]['model']['filter'].to(device)

                            # forward
                            output = dec.forward(centroid_i)

                            # edit
                            q2p = edit2_img * (1 - filter_i_edit) + output

                            # reconstruct
                            p2p = rec_img * (1 - filter_i_rec) + output

                            q2p_loss = get_mse_loss(edit1_img.float(), q2p.float())
                            q2p_loss = q2p_loss.to(device)

                            p2p_loss = get_mse_loss(rec_img.float(), p2p.float())
                            p2p_loss = p2p_loss.to(device)

                            loss = q2p_loss + p2p_loss
                            loss = loss.to(device)
                            loss_list.append(loss)
                    
                    elif 'and' in lesson.split():
                        bs = batch_size*3
                        rec_name_sim, rec_img_sim, _ , _ = dt.get_paired_batches(attr, lesson, bs)
                        edit_name_sim, edit_img_sim, edit_name_diff , edit_img_diff = dt.get_paired_batches(attr, lesson, bs)
                        loss_list = list()

                        for i in range(0,bs,3):
                            
                            with torch.no_grad():
                                # load sample
                                rec_name = rec_name_sim[i]
                                rec_img = clip_model.encode_image(rec_img_sim[i].unsqueeze(0).to(device))

                                edit1_name = edit_name_sim[i+2]
                                edit1_img = clip_model.encode_image(edit_img_sim[i].unsqueeze(0).to(device))

                                edit2_name = edit_name_diff[i+2]
                                edit2_img = clip_model.encode_image(edit_img_diff[i+2].unsqueeze(0).to(device))

                                # get second attribute for edit: lesson_filter
        
                                attrs = [attr.split(' ')[0], attr.split(' ')[2]]
                                attrs_idx = []
                                for a in attrs:
                                    attrs_idx.append(types_learning.index(a))
                                instances = edit2_name.split('_')[0:3]
                                filt_name = instances[attrs_idx[0]]+' and '+instances[attrs_idx[1]]

                                # load centroid
                                centroid_i = memory[lesson]['centroid'].float().to(device)

                                # load filters
                                filter_i_edit = memory[filt_name]['model']['filter'].to(device)

                                filter_i_rec = memory[lesson]['model']['filter'].to(device)

                            # forward
                            output = dec.forward(centroid_i)

                            # edit
                            q2p = edit2_img * (1 - filter_i_edit) + output

                            # reconstruct
                            p2p = rec_img * (1 - filter_i_rec) + output

                            q2p_loss = get_mse_loss(edit1_img.float(), q2p.float())
                            q2p_loss = q2p_loss.to(device)

                            p2p_loss = get_mse_loss(rec_img.float(), p2p.float())
                            p2p_loss = p2p_loss.to(device)

                            loss = q2p_loss + p2p_loss
                            loss = loss.to(device)
                            loss_list.append(loss)

                    elif 'or' in lesson.split():
                        bs = batch_size*3
                        rec_name_sim, rec_img_sim, _ , _ = dt.get_paired_batches(attr, lesson, bs)
                        edit_name_sim, edit_img_sim, edit_name_diff , edit_img_diff = dt.get_paired_batches(attr, lesson, bs)
                        loss_list = list()

                        for i in range(0,bs,3):
                            
                            with torch.no_grad():
                                # load sample
                                # reconstruct
                                rec_name_1 = rec_name_sim[i]
                                rec_img_1 = clip_model.encode_image(rec_img_sim[i].unsqueeze(0).to(device))

                                rec_name_2 = rec_name_sim[i+1]
                                rec_img_2 = clip_model.encode_image(rec_img_sim[i+1].unsqueeze(0).to(device))

                                rec_name_3 = rec_name_sim[i+2]
                                rec_img_3 = clip_model.encode_image(rec_img_sim[i+2].unsqueeze(0).to(device))

                                # edit
                                # es
                                # red or cone

                                # red_metal_cone 
                                # red_metal_teapot 
                                # green_metal_cone

                                # green_metal_cube - green or cube + out => red_metal_cone
                                # green_metal_teapot - green or teapot + out => red_metal_teapot
                                # green_metal_spot - spot + out => green_metal_cone

                                edit1_name_1 = edit_name_sim[i] # both
                                edit1_img_1 = clip_model.encode_image(edit_img_sim[i].unsqueeze(0).to(device))

                                edit1_name_2 = edit_name_sim[i+1] # first
                                edit1_img_2 = clip_model.encode_image(edit_img_sim[i+1].unsqueeze(0).to(device))

                                edit1_name_3 = edit_name_sim[i+2] # second
                                edit1_img_3 = clip_model.encode_image(edit_img_sim[i+2].unsqueeze(0).to(device))

                                ###
                                
                                edit2_name_1 = edit_name_diff[i]
                                edit2_img_1 = clip_model.encode_image(edit_img_diff[i].unsqueeze(0).to(device))

                                edit2_name_2 = edit_name_diff[i+1]
                                edit2_img_2 = clip_model.encode_image(edit_img_diff[i+1].unsqueeze(0).to(device))

                                edit2_name_3 = edit_name_diff[i+2]
                                edit2_img_3 = clip_model.encode_image(edit_img_diff[i+2].unsqueeze(0).to(device))

                                names_edit_diff = [edit2_name_1, edit2_name_2, edit2_name_3]

                                # get second attribute for edit: lesson_filter
                                
                                attrs = [attr.split(' ')[0], attr.split(' ')[2]]
                                attrs_idx = []
                                for a in attrs:
                                    attrs_idx.append(types_learning.index(a))
            
                                filt_names = []
                                for name in names_edit_diff:
                                    instances = name.split('_')[0:3]
                                    filt_names.append(instances[attrs_idx[0]]+' or '+instances[attrs_idx[1]])
                                   

                                # load centroid
                                centroid_i = memory[lesson]['centroid'].float().to(device)

                                # load filters
                                filter_i_edit_1 = memory[filt_names[0]]['model']['filter'].to(device)
                                filter_i_edit_2 = memory[filt_names[1]]['model']['filter'].to(device)
                                filter_i_edit_3 = memory[filt_names[2]]['model']['filter'].to(device)

                                filter_i_rec = memory[lesson]['model']['filter'].to(device)

                            # forward
                            output = dec.forward(centroid_i)
                            # edit
                            q2p_1 = edit2_img_1 * (1 - filter_i_edit_1) + output
                            q2p_2 = edit2_img_2 * (1 - filter_i_edit_2) + output
                            q2p_3 = edit2_img_3 * (1 - filter_i_edit_3) + output

                            # edit to do

                            # green_metal_cube - green or cube + out => red_metal_cone 
                            # green_metal_cube - green or cube + out => red_metal_cube
                            # green_metal_cube - green or cube + out => green_metal_cone

                            # reconstruct
                            p2p_1 = rec_img_1 * (1 - filter_i_rec) + output 
                            p2p_2 = rec_img_2 * (1 - filter_i_rec) + output  
                            p2p_3 = rec_img_3 * (1 - filter_i_rec) + output 

                            # 111 red_metal_cone   - red or cone + out => red_metal_cone  
                            # 110 red_metal_teapot - red or cone + out => red_metal_teapot
                            # 011 green_metal_cone - red or cone + out => green_metal_cone 

            
                            q2p_loss_1 = get_mse_loss(edit1_img_1.float(), q2p_1.float())
                            q2p_loss_1 = q2p_loss_1.to(device)
                            q2p_loss_2 = get_mse_loss(edit1_img_2.float(), q2p_2.float())
                            q2p_loss_2 = q2p_loss_2.to(device)
                            q2p_loss_3 = get_mse_loss(edit1_img_3.float(), q2p_3.float())
                            q2p_loss_3 = q2p_loss_3.to(device)


                            p2p_loss_1 = get_mse_loss(rec_img_1.float(), p2p_1.float())
                            p2p_loss_1 = p2p_loss_1.to(device)
                            p2p_loss_2 = get_mse_loss(rec_img_2.float(), p2p_2.float())
                            p2p_loss_2 = p2p_loss_2.to(device)
                            p2p_loss_3 = get_mse_loss(rec_img_3.float(), p2p_3.float())
                            p2p_loss_3 = p2p_loss_3.to(device)

                            loss = q2p_loss_1 + q2p_loss_2 + q2p_loss_3 + p2p_loss_1 + p2p_loss_2 + p2p_loss_3
                            loss = loss.to(device)
                            loss_list.append(loss)

                    elif 'not' in lesson.split():
                        rec_name_sim, rec_img_sim, rec_name_diff, rec_img_diff = dt.get_paired_batches(attr, lesson, bs)
                        edit_name_sim, edit_img_sim, edit_name_diff , edit_img_diff = dt.get_paired_batches(attr, lesson, bs)
                        loss_list = list()

                        for i in range(0,batch_size):
                            
                            with torch.no_grad():
                                # load sample
                                # not red
                                rec1_name = rec_name_sim[i] # not red
                                rec1_img = clip_model.encode_image(rec_img_sim[i].unsqueeze(0).to(device))

                                rec2_name = rec_name_diff[i] # red
                                rec2_img = clip_model.encode_image(rec_img_diff[i].unsqueeze(0).to(device))

                                edit1_name = edit_name_sim[i] # not red
                                edit1_img = clip_model.encode_image(edit_img_sim[i].unsqueeze(0).to(device))

                                edit2_name = edit_name_diff[i] # red
                                edit2_img = clip_model.encode_image(edit_img_diff[i].unsqueeze(0).to(device))

                                # load centroid
                                centroid_i = memory[lesson]['centroid'].float().to(device)

                                # load filters
                                filter_i_edit = memory[lesson.split(' ')[1]]['model']['filter'].to(device)

                                filter_i_rec = memory[lesson]['model']['filter'].to(device)

                            # forward
                            output = dec.forward(centroid_i)

                            # edit
                            q2p = edit2_img * (1 - filter_i_edit) + output # red - red + not red ---> max: sim(non red and out)

                            # reconstruct
                            p2p = rec1_img * (1 - filter_i_rec) + output # not red - not red + not red ---> max: sim(not red and out))

                            q2p_loss = get_mse_loss(edit1_img.float(), q2p.float())
                            q2p_loss = q2p_loss.to(device)

                            p2p_loss = get_mse_loss(rec1_img.float(), p2p.float())
                            p2p_loss = p2p_loss.to(device)

                            loss = q2p_loss + p2p_loss
                            loss = loss.to(device)
                            loss_list.append(loss)

                    stacked_loss = torch.stack(loss_list)
                    mean_loss = torch.mean(stacked_loss)
                    wandb.log({"loss": mean_loss})
                    # backward
                    optimizer.zero_grad()
                    mean_loss.backward()
                    optimizer.step()
                    
                    # print stats
                    if (round+1) % 10 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, epochs, round+1, 100, loss.item())) 
                        
            memory[lesson]['decoder'] = dec.to('cpu').state_dict()
            # save decoder
            with open(os.path.join(out_path, model_name), 'wb') as handle:
                pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
            wandb.finish()  # Finish W&B run when training is complete


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--in_path', '-i',
                help='Data input path', required=True)
    argparser.add_argument('--out_path', '-o',
                help='Model memory output path', required=True)
    argparser.add_argument('--model_name', '-n', default='mem_decoder_logic_small_mse.pickle',
                help='Best model memory to be saved file name', required=False)
    argparser.add_argument('--memory_path', '-m',
                help='Memory input path', required=True)

    args = argparser.parse_args()
   
    train_decoder(args.in_path, args.out_path, args.model_name, args.memory_path, clip_model, preprocess,
                    'train/', bn_train, ['rgba'], dec_dic_train_logical, all_vocabs,  
                    epochs=2, lr=0.001, batch_size=108, latent_dim=16)
