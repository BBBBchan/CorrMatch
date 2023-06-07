import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, asnumpy, parse_shape

a = torch.randn([4, 21, 65, 65])
b = torch.randn([4, 21, 65, 65])
c = torch.randn([4, 21, 65, 65])
h, w = a.shape[-2:]
h_local, w_local = 3, 3
for i in torch.arange(h - 2) + 1:
    for j in torch.arange(w - 2) + 1:
        a_local = a[:, :, i - 1:i + 2, j - 1:j + 2]
        a_local = rearrange(a_local, 'n c h w -> n c (h w)')
        b_local = b[:, :, i - 1:i + 2, j - 1:j + 2]
        b_local = rearrange(b_local, 'n c h w -> n c (h w)')
        corr_local = torch.matmul(a_local.transpose(1, 2), b_local) / torch.sqrt(torch.tensor(a_local.shape[1]).float())
        print(corr_local.shape)
        exit()
        c_local = c[:, :, i - 1:i + 2, j - 1:j + 2]
        c_local = rearrange(c_local, 'n c h w -> n c (h w)')
        out_local = rearrange(torch.matmul(c_local, F.softmax(corr_local, dim=-1)), 'n c (h w) -> n c h w', h=h_local,
                              w=w_local)
        print(out_local.shape)
