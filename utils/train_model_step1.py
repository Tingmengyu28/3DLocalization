# official modules
import os
import time
import math
import json
import numpy as np
from collections import defaultdict
import torch
from torch.cuda.amp import autocast, GradScaler
from utils.loss import calculate_loss
from utils.helper import print_log, print_metrics, save_checkpoint, print_time, get_cond


def train_model_v2(model,optimizer,scheduler,device,training_generator,validation_generator,log,opt):

    learning_results = defaultdict(list)
    max_epoch = opt.max_epoch
    steps_train = math.ceil(opt.ntrain / opt.batch_size / opt.world_size)
    steps_val = math.ceil(opt.nval / opt.batch_size / opt.world_size)
    params_val = {'batch_size': opt.batch_size, 'shuffle': False}
    path_save = opt.save_path

    # loss function
    calc_loss = torch.nn.MSELoss()
    scaler = GradScaler()

    if opt.resume:
        checkpoint = opt.save_path + '/step1' + opt.checkpoint_path + '/' + opt.checkpoint
        start_epoch = torch.load(checkpoint)['epoch']
        end_epoch = max_epoch

        # load all recorded metrics in checkpoint
        tmp_path = os.path.dirname(checkpoint)
        with open(os.path.join(tmp_path, 'learning_results.json'), 'r') as handle:
            learning_results = json.load(handle)

        if opt.rank==0:
            print_log('Total epoch in checkpoint: {}, continue from epoch {} and load loss\n'.format(len(learning_results['train_mseloss']),start_epoch),log)
        for key in learning_results.keys():
            try:
                learning_results[key] = learning_results[key][:start_epoch]
            except:
                pass

        # initialize validation set best loss and jaccard
        best_val_loss = np.min(learning_results['val_mseloss'])

    else:
        # start from scratch
        start_epoch, end_epoch = 0, max_epoch
        learning_results = {'train_mseloss': [], 'val_mseloss': [], 'val_max': [], 'val_sum': [], 'steps_per_epoch': steps_train}
        best_val_loss = float('Inf')

    # starting time of training
    train_start_time = time.time()
    not_improve = 0

    for epoch in np.arange(start_epoch, end_epoch):

        # starting time of current epoch
        epoch_start_time = time.time()

        if opt.rank == 0:
            print_log('Epoch {}/{}'.format(epoch+1, end_epoch), log)
            print_log('-' * 10, log)
            print_log(time.ctime(time.time()), log)
            print_log('lr '+str(optimizer.param_groups[0]['lr']),log)

        # training phase
        model.train()

        with torch.set_grad_enabled(True):
            for batch_ind, (inputs, targets, target_ims, fileids) in enumerate(training_generator):

                inputs = inputs.to(device)
                targets = targets.to(device)
                target_ims = target_ims.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                mseloss = calc_loss(outputs.float(), target_ims)

                scaler.scale(mseloss).backward()
                scaler.step(optimizer)
                scaler.update()

                if opt.rank==0:
                    print('Epoch {}/{} Train {}/{} mseloss: {:.4f} MaxOut: {:.2f}'.format( \
                    epoch+1,end_epoch,batch_ind+1,steps_train,mseloss,outputs.max()))
                    # img_mse {:.4f}  metric['img_mse'], fileid:%s scheduler.get_lr()[0], optimizer.param_groups[0]['lr'],ids:{}, fileids

        if opt.scheduler == 'StepLR':
            scheduler.step()

        # record training loss and jaccard results
        mean_train_loss = (mseloss/opt.ntrain)
        learning_results['train_mseloss'].append(mean_train_loss.tolist())

        # validation
        model.eval()

        with torch.set_grad_enabled(False):
            for batch_ind, (inputs, targets, target_ims, fileids) in enumerate(validation_generator):

                inputs = inputs.to(device)
                targets = targets.to(device)
                target_ims = target_ims.to(device)

                # forward
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                val_mseloss = calc_loss(outputs.float(), target_ims)

                if opt.rank==0:
                    print('Epoch {}/{} Val {}/{} mseloss:{:.4f} MaxOut:{:.2f}'.format( \
                        epoch+1,end_epoch, batch_ind+1,steps_val, val_mseloss, outputs.max()))
                # img_mse {:.4f}  metric['img_mse']    ids {} , fileids

        # record validation loss and jaccard results
        mean_val_loss = (val_mseloss/opt.nval)
        learning_results['val_mseloss'].append(mean_val_loss.tolist())

        if not opt.scheduler == 'StepLR':
            scheduler.step()

        # sanity check: record maximal value and sum of last validation sample
        max_last = outputs.max()
        sum_last = outputs.sum()/params_val['batch_size']
        learning_results['val_max'].append(max_last.tolist())
        learning_results['val_sum'].append(sum_last.tolist())

        # save checkpoint
        if opt.rank==0:
            if epoch%opt.save_epoch == opt.save_epoch - 1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(path_save, 'step1', 'checkpoint_'+str(epoch+1)))

        # save checkpoint for best val loss
        if mean_val_loss < (best_val_loss - 1e-4):
            if opt.rank==0:
                # print an update and save the model weights
                print_log('Val loss improved from %.4f to %.4f, saving best model...'% (best_val_loss, mean_val_loss), log)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(path_save, 'step1', 'checkpoint_best_loss'))
            # change minimal loss and init stagnation indicator
            best_val_loss = mean_val_loss
            not_improve = 0
        else:
            # update stagnation indicator
            not_improve += 1
            if opt.rank==0:
                print_log('Val loss not improve by %d epochs, best val loss: %.4f'% (not_improve,best_val_loss), log)

        epoch_time_elapsed = time.time() - epoch_start_time
        if opt.rank==0:
            print_log('Max test last: %.2f, Sum test last: %.2f' %(max_last, sum_last), log)
            print_log('{}, Epoch complete in {}\n'.format(time.ctime(time.time()),print_time(epoch_time_elapsed)), log)

            # save all records for latter visualization
            with open(os.path.join(path_save, 'step1', 'learning_results.json'), 'w') as handle:
                json.dump(learning_results, handle)

        # if no improvement for more than 15 epochs, break training
        if not_improve >= 15 or optimizer.param_groups[0]['lr']<1e-7:
            break

    # measure time that took the model to train
    train_time_elapsed = time.time() - train_start_time
    if opt.rank==0:
        print_log('Training complete in {}'.format(print_time(train_time_elapsed)), log)
        print_log('Best Validation Loss: {:6f}'.format(best_val_loss), log)

        learning_results['last_epoch_time'], learning_results['training_time'] = epoch_time_elapsed, train_time_elapsed
        with open(os.path.join(path_save, 'step1','learning_results.json'), 'w') as handle:
            json.dump(learning_results, handle)