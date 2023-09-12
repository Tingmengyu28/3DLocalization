# official modules
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from collections import deque
# self-defined modules
from utils.networks import *


def init_DDP(opt):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='123457'
    gpus = [g.strip() for g in opt.gpu_number.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[opt.rank]
    # dist.init_process_group('NCCL',rank=opt.rank,world_size=opt.world_size)
    # dist.init_process_group('GLOO',rank=opt.rank,world_size=opt.world_size)


class Logger(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    #self.writer = tf.summary.FileWriter(log_dir)
    self.writer = SummaryWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    # tf.summary.scalar(tag, value, step = step)
    self.writer.add_scalar(tag, value, step)

  def scalars_summary(self, tag, tag_scalar_dict, step):
        self.writer.add_scalars(tag, tag_scalar_dict, step)


def print_log(print_string, log, arrow=True):
    if arrow:
        print("---> {}".format(print_string))
        log.write('---> {}\n'.format(print_string))
        log.flush()
    else:
        print("{}".format(print_string))
        log.write('{}\n'.format(print_string))
        log.flush()


def load_labels(label_txt_path):
    label_raw = np.loadtxt(label_txt_path)
    if len(label_raw.shape) == 1:
        label_raw = np.expand_dims(label_raw, axis=0)
    labels = {}
    for i in range(len(label_raw)):
        if not str(int(label_raw[i,0])) in labels.keys():
            labels[str(int(label_raw[i,0]))] = label_raw[i,1:5]
            continue
        labels[str(int(label_raw[i,0]))]=np.c_[labels[str(int(label_raw[i,0]))], label_raw[i,1:5]]
    for i in labels.keys():
        if len(labels[i].shape)==2:
            labels[i] = np.expand_dims(labels[i].T,0) # [1,n,3]
        elif len(labels[i].shape)==1: # labels[i] = 3*n, n is number of source points,
            labels[i] = np.expand_dims(labels[i].T,0) # [1,3]
            labels[i] = np.expand_dims(labels[i],0) # [1,1,3]

    return labels


def build_model(opt):
    if opt.model_use == 'cnn':
        model = LocalizationCNN(opt)
    elif opt.model_use == 'cnn_residual':
        # 0808 difference with concatim is
        # (1) out = layer(out) + out -> residual conv layer
        # (2) deconv1 and deconv2 with +out or not
        model = ResLocalizationCNN(opt)
    elif opt.model_use == 'deq':
        model = DEQModel(opt)
    elif opt.model_use == 'dncnn':
        model = DnCNN(opt)
    elif opt.model_use == 'at_gt':
        model = AT_GT(opt)

    return model


def print_metrics(metrics,epoch_samples,phase,log):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))

    print_log("{}: {}".format(phase, ", ".join(outputs)),log)


# checkpoint saver for model weights and optimization status
def save_checkpoint(state, filename):
    torch.save(state, filename)


def print_time(epoch_time_elapsed):
    str = '{:.0f}h {:.0f}m {:.0f}s'.format(
        epoch_time_elapsed // 3600, 
        np.floor((epoch_time_elapsed / 3600 - epoch_time_elapsed // 3600)*60), 
        epoch_time_elapsed % 60)

    return str


def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

def dis_centroid_pf(p1, p2):
    return (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]))

def deepFirstSearch(index, xyz):
    p = xyz[index, :]

def equal_to_zero(array, valid_index):
    for i in valid_index:
        if array[i] != 0:
            return False
    return True


def BFS(xyz_rec, conf_rec, valid_points, scaling_factor):
    queue = deque()
    visited = set()
    start_point = int(np.argmax(conf_rec))
    queue.append(start_point)
    visited.add(start_point)

    while queue:
        p = queue.popleft()
        for i in valid_points:
            discp = dis_centroid_pf(xyz_rec[p, :], xyz_rec[i, :])
            if discp[0] <= 2 * scaling_factor and discp[1] <= 2 * scaling_factor and discp[2] <= 2 * scaling_factor and conf_rec[i] != 0:
                if i not in visited:
                    queue.append(i)
                    visited.add(i)
    return visited


def centroid(points, mass):
    x = y = z = 0
    m = sum(mass)
    for i in range(len(points)):
        x += points[i][0] * mass[i]
        y += points[i][1] * mass[i]
        z += points[i][2] * mass[i]
    return [x/m, y/m, z/m]


def get_cond(cs_ratio, sigma, cond_type, device):
    para_noise = sigma / 5.0
    if cond_type == 'org_ratio':
        para_cs = cs_ratio / 100.0
    else:
        para_cs = cs_ratio * 2.0 / 100.0

    para_cs_np = np.array([para_cs])
    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)
    para_cs = para_cs.to(device)

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.to(device)
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)

    return para


def get_precision_recall(gt_points, pred_points):
    tp = 0
    for gt_point in gt_points:
        min_distance = 1e5
        min_dx, min_dy, min_dz = 96, 96, 40
        for pred_point in pred_points:
            dx = abs(gt_point[0] - pred_point[0])
            dy = abs(gt_point[1] - pred_point[1])
            dz = abs(gt_point[2] - pred_point[2])
            if dx + dy + dz <= min_distance:
                min_dx = dx
                min_dy = dy
                min_dz = dz
                min_distance = dx + dy + dz
        if min_dx <= 2 and min_dy <= 2 and min_dz <= 2:
            tp += 1

    return (tp / len(pred_points), tp / len(gt_points))