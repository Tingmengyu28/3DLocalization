import scipy.io as io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.autograd as autograd
import numpy as np
import pandas as pd

# path = './Data/A_41slices.mat'
# path0 = './Data/train_images/observed/im0.mat'
# path1 = './Data/train_images/noiseless/I28.mat'
# path_loc = './Data/results/test_results/loc.csv'
# path_loc_pp = './Data/results/test_results/post_processes/loc.csv'
# A = io.loadmat(path)['A']
# im = io.loadmat(path1)['I0']
# pred = pd.read_csv(path_loc, sep=',', header='infer')
# pred_pp = pd.read_csv(path_loc_pp, sep=',', header='infer')
# f = open('./Data/results/label.txt')

# x = torch.zeros((96, 96, 41))
# x[45, 36, 15] = 1
# x[76, 15, 37] = 1

# identity = 28
# xp, yp, zp = [], [], []
# # for i in range(pred.shape[0]):
# #     if pred.loc[i, 'frame'] == identity and pred.loc[i, 'intensity'] >= 0.04:
# #         xp.append(pred.loc[i, 'x'])
# #         yp.append(pred.loc[i, 'y'])
# #         zp.append(pred.loc[i, 'z'])
# for i in range(pred_pp.shape[0]):
#     if pred_pp.loc[i, 'frame'] == identity and pred_pp.loc[i, 'intensity'] >= 0.04:
#         xp.append(pred_pp.loc[i, 'x'])
#         yp.append(pred_pp.loc[i, 'y'])
#         zp.append(pred_pp.loc[i, 'z'])


# xt, yt, zt = [], [], []
# for line in f:
#     st = line.split(' ')
#     if st[0] == str(identity):
#         xt.append(float(st[1]))
#         yt.append(float(st[2]))
#         zt.append(float(st[3]))

# fig=plt.figure(figsize=(12, 6), facecolor='w')
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(xp, yp, zp, alpha=0.3, c="#FF0000")
# ax1.scatter(xt, yt, zt, alpha=0.3, c="#0000FF")
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.imshow(im)
# plt.show()
print(torch.tensor([[1], [0.5]], dtype=torch.float32))