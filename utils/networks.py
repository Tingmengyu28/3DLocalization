# Import modules and libraries
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.functional import interpolate
from lib.solvers import anderson, broyden, normal
from lib.jacobian import jac_loss_estimate
import math
import numpy as np

###############################################
## blocks for all network
################################################

class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


class ResConv2DLeakyReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, negative_slope):
        super(ResConv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)

        # if self.in_channels == self.out_channels:
        out += residual

        return out


class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=4):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv3 = nn.Conv2d(n_inner_channels, n_inner_channels * 2, kernel_size, padding=kernel_size//2, bias=False)
        self.conv4 = nn.Conv2d(n_inner_channels * 2, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.norm4 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm5 = nn.GroupNorm(num_groups, n_inner_channels * 2)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        y = self.norm4(F.relu(self.conv4(self.norm5(F.relu(self.conv3(y))))))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))


########################################
# network structure below
########################################

# 0721 replace plain conv layer by residual layer
class ResLocalizationCNN(nn.Module):
    def __init__(self, opt):
        super(ResLocalizationCNN, self).__init__()
        self.opt = opt

        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)

        self.layer3 = ResConv2DLeakyReLUBN(64, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = ResConv2DLeakyReLUBN(64, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = ResConv2DLeakyReLUBN(64, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = ResConv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.deconv2 = ResConv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, opt.D, 3, 1, 1, 0.2)
        self.layer8 = ResConv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer9 = ResConv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(opt.D, opt.D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=opt.scaling_factor)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        im = im.view(self.opt.batch_size, -1, self.opt.H, self.opt.W)

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        out = self.layer2(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.layer3(out)  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)

        if self.opt.upsampling_factor == 2:
            # upsample by 2 in xy
            out = interpolate(out, scale_factor=2)
        elif self.opt.upsampling_factor == 4:
            out = interpolate(out, scale_factor=4)
        
        out = self.deconv1(out)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


# original Localization architecture
class LocalizationCNN(nn.Module):
    def __init__(self, opt):
        super(LocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)

        self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)

        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, opt.D, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(opt.D, opt.D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=opt.scaling_factor)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):  # [4,1,96,96]

        # extract multi-scale features
        im = self.norm(im) # [4,1,96,96]
        out = self.layer1(im)  # [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer2(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        features = torch.cat((out, im), 1) # [4,65,96,96]
        out = self.layer3(features) + out  # [4,64,96,96] -> +out = [4,64,96,96]
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        out = self.dropout(out)
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        # out = features
        out = self.deconv1(out)
        # out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


class DnCNN(nn.Module):
    def __init__(self, opt):
        super(DnCNN, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.convBnRelu1 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.convBnRelu2 = Conv2DLeakyReLUBN(64, 64, 3, 2, 2, 0.2)
        self.convBnRelu3 = Conv2DLeakyReLUBN(64, 64, 3, 4, 4, 0.2)
        self.convBnRelu4 = Conv2DLeakyReLUBN(64, 64, 3, 8, 8, 0.2)
        self.convBnRelu5 = Conv2DLeakyReLUBN(64, 64, 3, 16, 16, 0.2)
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)


    def forward(self, im):
        out = F.relu(self.conv1(im))
        out = self.convBnRelu1(out)
        out = self.convBnRelu2(out)
        out = self.convBnRelu3(out)
        out = self.convBnRelu4(out)
        out = self.convBnRelu5(out)
        out = F.relu(self.conv2(out))
        return out


class AT_GT(nn.Module):
    def __init__(self, opt):
        super(AT_GT, self).__init__()
        self.opt = opt
    
    def AnscombeTransform(self, input):
        return 2 * (input + 0.375)**0.5

    def inverse_AnscombeTransform(self, input):
        return 0.25 * input**2 + 0.25 * 1.5**0.5 * input**(-1) - 11 / 8 * input**(-2) + 0.625 * 1.5**0.5 * input**(-3) - 0.125

    def gaussian_filter(self, img, K_size=3, sigma=1.3):
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
    
        img = img.numpy()
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    
        K = np.zeros((K_size, K_size), dtype=np.float)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    
        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()
        tmp = out.copy()
    
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    
        out = np.clip(out, 0, 255)
        out = out[pad: pad + H, pad: pad + W].astype(np.float32)
        return out

    def forward(self, im):
        im = torch.squeeze(im, 0)
        at_im = self.AnscombeTransform(im)
        at_im = at_im.to('cpu')
        at_im = self.gaussian_filter(at_im)
        at_im = torch.tensor(at_im)
        at_im = at_im.to('cuda')
        out = self.inverse_AnscombeTransform(at_im)
        return out

    
class DEQFixedPoint(nn.Module):
    def __init__(self, opt, f, solver):
        super(DEQFixedPoint, self).__init__()
        self.opt = opt
        self.f = f
        self.solver = solver
    
    def forward(self, x):
        z0 = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x), z0, threshold=self.opt.f_thres)   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.opt.train_or_test == 'train':
            z_star.requires_grad_()
            new_z_star = self.f(z_star, x)
            new_z_star.requires_grad_()
            
            # Jacobian-related computations, see additional step above. For instance:
            # jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                f = lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad
                new_grad = self.solver(f, torch.zeros_like(grad), threshold=self.opt.b_thres)
                return new_grad
            self.hook = new_z_star.register_hook(backward_hook)
        return new_z_star


class DEQModel(nn.Module):
    def __init__(self, opt):
        super(DEQModel, self).__init__()

        self.opt = opt
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, bias=True, padding=1)
        self.norm1 = nn.BatchNorm2d(4)
        self.deq = DEQFixedPoint(opt, ResNetLayer(4, 32), normal)
        self.norm2 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, bias=True, padding=1)
        self.norm3 = nn.BatchNorm2d(1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.deq(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out