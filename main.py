# official modules
import os
import time
import json
import argparse
import numpy as np
import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from lib import radam
from utils.helper import Logger
import torch.optim as optim
# self-defined module
from utils.helper import init_DDP, print_log, load_labels, build_model
from utils.data import dataloader
from utils.train_model import train_model, train_model_step1
from utils.test_model import test_model


def learn_localization(opt, rank=0, world_size=1):
    opt.rank = rank
    opt.world_size = world_size

    if opt.train_or_test == 'train':
        step_mode = opt.step_mode
        # split data to train and validation set
        tmp_train = np.arange(0, 9000, 1).tolist()
        tmp_val = np.arange(9000, 10000, 1).tolist()
        train_IDs = [str(i) for i in tmp_train]
        val_IDs = [str(i) for i in tmp_val]

        opt.partition = {'train': train_IDs, 'valid': val_IDs}
        opt.ntrain, opt.nval = len(train_IDs), len(val_IDs)

        # calculate zoom ratio of z-axis
        opt.pixel_size_axial = (
            opt.zeta[1] - opt.zeta[0] + 1 + 2*opt.clear_dist) / opt.D

        t_simple = 'model_' + time.strftime('%m%d')

        opt.save_path = os.path.join(opt.save_path, 'step' + step_mode, t_simple)
        os.makedirs(opt.save_path, exist_ok=True)

        log = open(os.path.join(opt.save_path, 'log_{}.txt'.format(time.strftime('%H%M'))), 'w')
        logger = Logger(os.path.join(opt.save_path, 'log_{}'.format(time.strftime('%m%d'))))

        print_log('setup_params:', log)
        for key, value in opt._get_kwargs():
            if not key == 'partition':
                print_log('{}: {}'.format(key, value), log)

        device = torch.device('cuda')

        # save setup parameters in results folder as well
        with open(os.path.join(opt.save_path, 'setup_params.json'), 'w') as handle:
            json.dump(opt.__dict__, handle, indent=2)

        # Load labels and generate dataset
        labels = load_labels(os.path.join(
            opt.data_path, 'observed', 'label.txt'))

        # Parameters for dataloaders
        params_train = {'batch_size': opt.batch_size,
                        'shuffle': True,  'partition': opt.partition['train']}
        params_val = {'batch_size': opt.batch_size,
                    'shuffle': False, 'partition': opt.partition['valid']}

        training_generator = dataloader(
            opt.data_path, labels, params_train, opt, num_workers=0)
        validation_generator = dataloader(
            opt.data_path, labels, params_val, opt, num_workers=0)

        # model
        model = build_model(opt)
        model.to(device)
            
        optimizer = optim.Adam(model.parameters(), lr=opt.initial_learning_rate)
        #### scheduler
        if opt.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0)
        elif opt.scheduler == 'inv_sqrt':
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and opt.warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > opt.warmup_step else step / (opt.warmup_step ** 1.5)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif opt.scheduler == 'dev_perf':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, min_lr=0)

        # Print Model layers and number of parameters
        print_log(model, log)
        print_log("number of parameters: {}\n".format(sum(param.numel() for param in model.parameters())), log)

        # if resume_training, continue from a checkpoint
        if opt.resume:
            checkpoint_path = './Data/results/step' + step_mode + opt.checkpoint_path + '/' + opt.checkpoint
            try:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass

        if opt.step_mode == 'all':
            train_model(model, optimizer, scheduler, device,
                        training_generator, validation_generator, log, logger, opt)
        elif opt.step_mode == '1':
            train_model_step1(model, optimizer, scheduler, device,
                        training_generator, validation_generator, log, opt)

    elif opt.train_or_test == 'test':
        checkpoint_path = './Data/results/step' + step_mode + opt.checkpoint_path + '/' + opt.checkpoint
        opt.device = 'cuda'
        if opt.postpro:
            opt.postpro_params = {'thresh': 0.05, 'radius': 1}

        time_start = time.time()
        os.makedirs(opt.save_path, 'step' + step_mode, exist_ok=True)
        opt.pixel_size_axial = (opt.zeta[1] - opt.zeta[0] + 1 + 2*opt.clear_dist) / opt.D

        # model testing
        model = build_model(opt)
        model.to('cuda')
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        model.eval()

        log = open(os.path.join(opt.save_path, 'step' + step_mode, 'test_results', 'log_{}.txt'.format(time.strftime('%H%M'))), 'w')
        print_log('setup_params -- test:', log)
        for key, value in opt._get_kwargs():
            if not key == 'partition':
                print_log('{}: {}'.format(key, value), log)

        # save setup parameters in results folder as well
        with open(os.path.join(opt.save_path, 'step' + step_mode, 'setup_params_test.json'), 'w') as handle:
            json.dump(opt.__dict__, handle, indent=2)

        test_model(opt, model, log)
        time_end = time.time()
        print(f'Time cost: {time_end-time_start}')
    else:
        print('no such process!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3d localization')
    parser.add_argument('--train_or_test',            type=str,
                        default='train',        help='train or test')
    parser.add_argument('--resume',
                        action='store_true', default=False)
    parser.add_argument('--gpu_number',               type=str,
                        default='1',              help='assign gpu')
    # data info
    parser.add_argument('--H',                        type=int,
                        default=96,          help='Height of image')
    parser.add_argument('--W',                        type=int,
                        default=96,          help='Width of image')
    parser.add_argument('--zeta',                     type=tuple,
                        default=(-20, 20),    help='min and mx zeta')
    parser.add_argument('--clear_dist',               type=int,
                        default=0,           help='safe margin for z axis')
    parser.add_argument('--D',                        type=int,
                        default=400,         help='num grid of zeta axis')
    parser.add_argument('--scaling_factor',           type=int,
                        default=800,         help='entry value for existence of pts')
    parser.add_argument('--upsampling_factor',        type=int,           default=4,
                        help='grid dim=H*upsampling_factor, W*upsampling_factor')
    
    # train info
    parser.add_argument('--model_use',
                        type=str,           default='deq')
    parser.add_argument('--step_mode',
                        type=str,           default='all')
    parser.add_argument('--postpro',                action='store_true',
                        default=False,       help='whether do post processing in dnn')
    parser.add_argument('--batch_size',               type=int,           default=8,
                        help='when training on multi GPU, is the batch size on each GPU')
    parser.add_argument('--initial_learning_rate',    type=float,
                        default=0.0005,      help='initial learning rate for adam')

    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                        help='optimizer to use.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='the number of steps to warm up the learning rate to its lr value')
    
    parser.add_argument('--lr_decay_per_epoch',       type=int,
                        default=10,          help='number of epochs learning rate decay')
    parser.add_argument('--lr_decay_factor',          type=float,
                        default=0.5,         help='lr decay factor')
    parser.add_argument('--max_epoch',                type=int,
                        default=30,          help='number of training epoches')
    parser.add_argument('--save_epoch',               type=int,
                        default=3,           help='save model per save_epoch')
    parser.add_argument('--f_thres', type = int, 
                        default=500)
    parser.add_argument('--b_thres', type = int, 
                        default=800)

    # test info
    parser.add_argument('--test_id_loc', type=str,
                        default='./Data/id_test.txt')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint_21', help='checkpoint to resume from')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/model_0413')
    parser.add_argument('--data_path', type=str,
                        default='./Data/train_images', help='path for train and val data')
    parser.add_argument('--save_path', type=str, default='./Data/results',
                        help='path for save models and results')

    opt = parser.parse_args()

    # gpu_number = len(opt.gpu_number.split(','))
    # mp.spawn(learn_localization,args=(gpu_number,opt),nprocs=gpu_number,join=True)
    learn_localization(opt)
