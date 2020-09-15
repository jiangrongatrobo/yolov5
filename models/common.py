# This file contains modules common to various models
import math
# import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)

class DynamicBatchNorm2d(nn.Module):
    """
    可变层数的BN, copy from once-for-all
    """
    SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()
        
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)
    
    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0
            
            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )
    
    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y

class ElasticPointConv(nn.Module):
    # by jiangrong
    def __init__(self, c1, c2, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ElasticPointConv, self).__init__()
        self.max_out_channel = c2
        self.in_channel = c1
        self.conv = nn.Conv2d(self.in_channel, self.max_out_channel
                            , 1, 0, 0, groups=1, bias=False)
        self.bn = DynamicBatchNorm2d(self.max_out_channel)
        self.act = nn.ReLU() if act else nn.Identity()
        self.real_output_channel = self.max_out_channel

    def forward(self, x):
        filters = self.conv.weight[:self.real_output_channel, :, :, :]
        return self.act(self.bn(F.conv2d(input=x
                                        , weight=filters
                                        , bias=None
                                        , stride=1
                                        , padding=0
                                        , groups=1
                                        )))

    def re_organize_middle_weights(self, sorted_idx):
        # 对输出权重sort
        self.conv.weight.data = torch.index_select(
                                self.conv.weight.data, 0, sorted_idx)
        adjust_bn_according_to_idx(self.bn, sorted_idx)
        return


class ElasticConv(nn.Module):
    # by jiangrong
    KERNEL_TRANSFORM_MODE = 1
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ElasticConv, self).__init__()
        self.kernel_size_list = k
        # for dynamic kernel size
        self.max_kernel_size = max(self.kernel_size_list)
        self.active_kernel_size = self.max_kernel_size
        self.padding = p
        self.stride = s
        self.max_input_channel = c1
        self.output_channel = c2
        # for dynamic channel size
        self.real_input_channel = self.max_input_channel
        self.conv = nn.Conv2d(self.max_input_channel, self.output_channel
                            , self.max_kernel_size, self.stride
                            , autopad(self.max_kernel_size, self.padding)
                        , groups=1, bias=False)
        self.bn = nn.BatchNorm2d(self.output_channel)
        self.act = nn.ReLU() if act else nn.Identity()
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7=>5=>3 ...
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                # 对卷机的参数变换, max_in * k * k
                scale_params['transform_%d_%d_matrix' % (ks_larger, ks_small)] = \
                        Parameter(torch.eye(ks_small * ks_small * c1)) # kernel shape: out x in
            for name, param in scale_params.items():
                self.register_parameter(name, param)

    def get_active_filter(self):
        start, end = sub_filter_start_end(self.max_kernel_size, self.active_kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end] # out, in, h, w
        # jiangrong: 要得到3x3kernel的话，先crop 7x7kernel中间的5x5, 然后对5x5做线性变换。同样的过程再转换5x5到3x3。
        if self.KERNEL_TRANSFORM_MODE is not None and self.active_kernel_size < self.max_kernel_size:
            start_filter = self.conv.weight  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= self.active_kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1) # out, in, k*k
                _input_filter = _input_filter.view(-1, _input_filter.size(1) * _input_filter.size(2)) # out, in*k*k
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('transform_%d_%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2) # out, in, k*k
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks) # out, in, k, k
                start_filter = _input_filter
            filters = start_filter
        
        # for dynamic input channel
        filters = filters[:, :self.real_input_channel, :, :]
        return filters

    def forward(self, x):
        filters = self.get_active_filter().contiguous()
        return self.act(self.bn(F.conv2d(input=x
                                        , weight=filters
                                        , bias=None
                                        , stride=self.stride
                                        , padding=autopad(self.active_kernel_size, self.padding)
                                        # , dilation=self.dilation
                                        , groups=1
                                        )))
    
    def expand_sort_index(self, sorted_index, expand_size):
        """
        from: sorted_index=[4,1,2,3], expand_size=3
        to: [4*3+0, 4*3+1, 4*3+2, 1*3+0, 1*3+1, 1*3+2, 2*3+0, 2*3+1, 2*3+2, 3*3+0, 3*3+1, 3*3+2]
        """
        expanded_sorted_index = []
        for index in sorted_index.detach().tolist():
            for shift in range(expand_size):
                expanded_sorted_index.append(index * expand_size + shift)
        expanded_sorted_index = torch.Tensor(expanded_sorted_index)
        return expanded_sorted_index

    def re_organize_middle_weights(self):
        # 对输入做sort
        # self.conv.weight.data.size(): out, in, height, weight
        importance = torch.sum(torch.abs(self.conv.weight.data), dim=(0, 2, 3))
        _, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv.weight.data = torch.index_select(
                                self.conv.weight.data, 1, sorted_idx)
        # change sort for kernel transform weights
        if self.KERNEL_TRANSFORM_MODE:
            for transform in self.__dict__.keys():
                if transform.startswith('transform_'):
                    _, src_ks, target_ks, _ = transform.split('_')
                    self.__dict__['transform'] = torch.index_select(
                                                self.__dict__['transform'], 1, expand_sort_index(sorted_idx))
        return sorted_idx

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.Hardswish() if act else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ElasticBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, k=[3,5], e=[0.3, 0.5, 0.7], g=1): # ch_in, ch_out, shortcut, groups, expansion
        super(ElasticBottleneck, self).__init__()
        self.input_channel = c1
        self.output_channel = c2
        self.expansion_ratio_list = e
        self.max_expansion_ratio = max(self.expansion_ratio_list)
        self.real_expansion_ratio = self.max_expansion_ratio
        self.max_mid_channel = int(self.output_channel * self.max_expansion_ratio)  # hidden channels
        self.kernel_size_list = k
        self.cv1 = ElasticPointConv(self.input_channel, self.max_mid_channel, True)
        self.cv2 = ElasticConv(self.max_mid_channel, self.output_channel, self.kernel_size_list, 1)
        self.add = shortcut and self.input_channel == self.output_channel

    def forward(self, x):
        real_mid_channel = int(self.output_channel * self.real_expansion_ratio)  # hidden channels
        self.cv1.real_output_channel = real_mid_channel
        self.cv2.real_input_channel = real_mid_channel
        a1 = self.cv1(x)
        a2 = self.cv2(a1)
        # print('===>', x.size(), self.add, a1.size(), a2.size())
        return x + a2 if self.add else a2
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)

class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
