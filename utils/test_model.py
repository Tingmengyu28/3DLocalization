# official modules
from shutil import copy2
import numpy as np
import time
import os
from collections import defaultdict
import csv
import torch
import torch.distributed as dist
# self-defined modules
from utils.loss import calculate_loss
from utils.data import dataloader
from utils.postprocess import Postprocess
from utils.helper import print_log, load_labels, print_time


def test_model(opt, model, log=None):

    imgs_path = opt.data_path
    save_path = opt.save_path + '/step' + opt.step_mode
    checkpoint_path = './Data/results/step' + opt.step_mode + opt.checkpoint_path + '/' + opt.checkpoint

    # copy setup params of train model into infer save folder
    path_train_result = os.path.dirname(checkpoint_path)
    copy2(os.path.join(path_train_result,'setup_params.json'),os.path.join(save_path,'setup_params_train.json'))

    if opt.rank==0:
        print_log('setup_params -- train:',log)
        for key,value in opt._get_kwargs():
            if not key == 'partition':
                print_log('{}: {}'.format(key,value),log)

    device= 'cuda'
    calc_loss = calculate_loss(opt)

    try:
        # read imgs with id in opt.test_id_loc if specific img list for test are chosen
        img_names = []
        with open(opt.test_id_loc,'r') as file:
            for line in file:
                img_names.append(line[:-1])
        if opt.rank==0:
            all_imgs = [x for x in os.listdir(imgs_path) if 'mat' in x and 'im' in x]
            print_log('{}/{} imgs are chosen to test in {}'.format(len(img_names),len(all_imgs),imgs_path), log)
            print_log(img_names,log)
    except:
        img_names = [x[2:-4] for x in os.listdir(imgs_path) if 'mat' in x and 'im' in x]

    # load label file if label exist
    label_existence = os.path.exists(os.path.join(imgs_path,'observed','label.txt'))
    if label_existence:
        copy2(os.path.join(imgs_path,'observed','label.txt'),os.path.join(save_path,'label.txt'))
        if opt.rank == 0:
            print_log('label exists!',log)
        labels = load_labels(os.path.join(imgs_path,'observed','label.txt'))

        params_test = {'batch_size': 1, 'shuffle': False, 'partition':img_names}
        data_generator = dataloader(imgs_path, labels, params_test, opt)

    postpro_module = Postprocess(opt)

    # time the entire dataset analysis
    tall_start = time.time()

    # process all experimental images
    model.eval()
    if opt.step_mode == 'all':
        metric = defaultdict(float)
        results = np.array(['frame', 'x', 'y', 'z', 'intensity'])
        results_bol = np.array(['frame', 'x', 'y', 'z', 'intensity'])
        pt = open(os.path.join(save_path, 'test_results','loss_{}.txt'.format(opt.rank)),'w')
        with torch.set_grad_enabled(False):
            for im_ind, (im_tensor, target, target_2d, fileid) in enumerate(data_generator):
                
                im_tensor = im_tensor.to(device)
                target = target.to(device)
                target_2d = target_2d.to(device)

                pred_volume = model(im_tensor)

                pt.write('{},{:.4f},{:.4f},{:.4f}\n'.format(fileid[0], metric['dice'], metric['kde'], metric['mse2d']))

                # post-process result to get the xyz coordinates and their confidence
                xyz_rec, conf_rec, xyz_bool = postpro_module(pred_volume)

                # if prediction is empty then set number for found emitters to 0
                # otherwise generate the frame column and append results for saving
                if xyz_rec is None:
                    nemitters = 0
                else:
                    nemitters = xyz_rec.shape[0]
                    frm_rec = (int(img_names[im_ind]))*np.ones(nemitters)
                    results = np.vstack((results, np.column_stack((frm_rec, xyz_rec, conf_rec))))
                    results_bol = np.vstack((results_bol, np.column_stack((frm_rec, xyz_bool, conf_rec))))

                if opt.rank==0:
                    if label_existence:
                        xyz_gt = np.squeeze(labels[fileid[0]])
                        print_log('Test sample {} Img{} found {:d}/{} emitters'.format(im_ind, fileid[0],nemitters,len(xyz_gt)),log)
                    else:
                        print_log('Test sample {} Img{} found {:d} emitters'.format(im_ind, fileid[0], nemitters),log)

        # print the time it took for the entire analysis
        tall_end = time.time() - tall_start
        if opt.rank==0:
            print_log('=' * 50,log, arrow=False)
            print_log('Analysis complete in {}'.format(print_time(tall_end)), log)
            print_log('=' * 50,log, arrow=False)

        # write the results to a csv file named "loc.csv" under the infer result save folder
        if opt.postpro == True:
            row_list = results.tolist()
            with open(os.path.join(save_path, 'test_results/post_processes','loc.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
            pt.close()

            row_list = results_bol.tolist()
            with open(os.path.join(save_path, 'test_results/post_processes', 'loc_bool.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
            pt.close()
        else:
            row_list = results.tolist()
            with open(os.path.join(save_path, 'test_results','loc.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
            pt.close()

            row_list = results_bol.tolist()
            with open(os.path.join(save_path, 'test_results', 'loc_bool.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
            pt.close()

        # dist.barrier()
        if opt.rank == 0:
            pt = open(os.path.join(save_path, 'test_results', 'loss.txt'),'w')
            for i in range(opt.world_size):
                with open(os.path.join(save_path, 'test_results','loss_{}.txt'.format(i)),'r') as tmpPt:
                    pt.writelines(tmpPt.readlines())
            pt.close()
        return xyz_rec, conf_rec
    
    elif opt.step_mode == '1':
        metric = defaultdict(float)
        pt = open(os.path.join(save_path, 'test_results','loss_{}.txt'.format(opt.rank)),'w')
        with torch.set_grad_enabled(False):
            for im_ind, (im_tensor, target, target_2d, fileid) in enumerate(data_generator):
                
                im_tensor = im_tensor.to(device)
                target = target.to(device)
                target_2d = target_2d.to(device)

                pred_im = model(im_tensor)
                mseloss = torch.nn.MSELoss(pred_im, target_2d)

                print_log('Test sample {} Img{} complete, MSELoss is {:.4f}'.format(im_ind, fileid[0], mseloss))

        tall_end = time.time() - tall_start
        if opt.rank==0:
            print_log('=' * 50,log, arrow=False)
            print_log('Analysis complete in {}'.format(print_time(tall_end)), log)
            print_log('=' * 50,log, arrow=False)

        return pred_im