# Import modules and libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.functional import interpolate
from lib.solvers import anderson, broyden, normal
from lib.jacobian import jac_loss_estimate
import math

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
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(n_channels, n_channels)
        self.norm3 = nn.GroupNorm(n_channels, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
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


# COAST unrolling network
class CPMB(nn.Module):
    '''Residual block with scale control
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, res_scale_linear, nf=32):
        super(CPMB, self).__init__()

        conv_bias = True
        # scale_bias = True
        # cond_dim = 2

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.res_scale = res_scale_linear
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]
        cond = cond[:, 0:1]
        cond_repeat = cond.repeat((content.shape[0], 1))
        out = self.act(self.conv1(content))
        out = self.conv2(out)
        res_scale = self.res_scale(cond_repeat)
        alpha1 = res_scale.view(-1, 32, 1, 1)
        out1 = out * alpha1
        return content + out1, cond
    

class BasicBlock(nn.Module):
    def __init__(self, res_scale_linear):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.head_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiTb, cond, block_size):
        x = x - self.lambda_step * x
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, block_size, block_size)

        x_mid = self.head_conv(x_input)
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)
        x_pred = x_input + x_mid

        x_pred = x_pred.view(-1, block_size * block_size)

        return x_pred
    

class COAST(nn.Module):
    def __init__(self, LayerNo = 20, nf = 32):
        super(COAST, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        scale_bias = True
        res_scale_linear = nn.Linear(1, nf, bias=scale_bias)

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, block_size=96):

        I = torch.eye(x.shape, dtype=x.dtype, device=x.device).detach()
        cond = torch.tensor((1, 0.5))

        PhiTb = x.clone()

        for i in range(self.LayerNo):
            x = self.fcs[i](x, I, PhiTb, cond, block_size)

        x_final = x

        return x_final


class COASTModel(nn.Module):
    def __init__(self, opt):
        super(COASTModel, self).__init__()
        self.opt = opt
        self.coast = COAST()
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64, opt.D, 3, 1, 1, 0.2)
        self.layer3 = ResConv2DLeakyReLUBN(opt.D, opt.D, 3, 1, 1, 0.2)
        self.layer4 = nn.Conv2d(opt.D, opt.D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=opt.scaling_factor)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im):
        out = self.coast(im)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.opt.upsampling_factor == 2:
            # upsample by 2 in xy
            out = interpolate(out, scale_factor=2)
        elif self.opt.upsampling_factor == 4:
            out = interpolate(out, scale_factor=4)

        out = self.dropout(out)
        out = self.layer4(out)
        out = self.pred(out)
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