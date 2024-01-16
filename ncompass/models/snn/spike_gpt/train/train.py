import torch.nn as nn

#TODO: Rewrite this if necessary, has been taken directly from ridgerchu/SpikeGPT
# source: src/model.py
from ncompass.internal.third_party.spikingjelly.clock_driven.surrogate import ATan as atan
from ncompass.internal.third_party.spikingjelly.clock_driven import neuron, surrogate
import math, os
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

try:    from deepspeed.ops.adam import FusedAdam
except: pass # some poor windows users cant install deepspeed

logger = logging.getLogger(__name__)

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 1024  # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["/home/ec2-user/nCompass/ncompass/internal/third_party/cuda/wkv_op.cpp", "/home/ec2-user/nCompass/ncompass/internal/third_party/cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = -torch.exp(w.float().contiguous())
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(model, args):  # fancy initialization of all lin & emb layer in the model
    print("\n[--> first run, init model params (very slow for large models) <--]")
    print("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == args.vocab_size and shape[1] == args.hidden_size:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == args.vocab_size and shape[1] == args.hidden_size:  # final projection?
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            # print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {name}")

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.hidden_size = config.hidden_size

        attn_sz = config.hidden_size

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = (layer_id / (config.num_hidden_layers - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / config.num_hidden_layers))  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, config.hidden_size)
            for i in range(config.hidden_size):
                x[0, 0, i] = i / config.hidden_size
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.hidden_size, attn_sz, bias=False)
        self.value = nn.Linear(config.hidden_size, attn_sz, bias=False)
        self.receptance = nn.Linear(config.hidden_size, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.hidden_size, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x):

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        sr, k, v = self.jit_func(x)

        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / config.num_hidden_layers))  # 1 to ~0

            x = torch.ones(1, 1, config.hidden_size)
            for i in range(config.hidden_size):
                x[0, 0, i] = i / config.hidden_size

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * config.hidden_size
        self.key = nn.Linear(config.hidden_size, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(hidden_sz, config.hidden_size, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


########################################################################################################
# The GPT Model with our blocks
########################################################################################################
class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lif1 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
                                            backend='torch',
                                            v_threshold=1.)
        self.lif2 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0),
                                            backend='torch',
                                            v_threshold=1.)
        # self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan(),step_mode='m',backend='cupy',detach_reset=True)
        # self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan(),step_mode='m',backend='cupy',detach_reset=True)
        self.dropout = nn.Dropout(0.03)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size)

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, 0)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.lif1(self.ffnPre(self.ln1(x)).permute(1, 0, 2)).permute(1, 0, 2)  # better in some cases
        else:
            x = x + self.lif1(self.att(self.ln1(x)).permute(1, 0, 2)).permute(1, 0, 2)
        x = x + self.lif2(self.ffn(self.ln2(x)).permute(1, 0, 2)).permute(1, 0, 2)
        x = self.dropout(x)

        return x

class SpikeGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)

        self.blocks = nn.Sequential(*[Block(config, i)
                                      for i in range(config.num_hidden_layers)])
        self.atan = atan()
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.hidden_size, config.RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.hidden_size, config.RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config)
        except:
            pass

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        if config.name_or_path != "":
            _model = torch.load(config.name_or_path, map_location=torch.device('cpu'))
            self.load_state_dict(_model)

        if config.use_cuda: self.cuda()

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        try:
            optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas,
                                  eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0,
                                  amsgrad=False)
        except:
            print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
            optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas,
                                         eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None):
        idx = idx.to(self.emb.weight.device)

        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.atan(self.emb(idx))
        x = self.blocks(x)
        x = self.ln_out(x)

        if self.config.RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / self.config.RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            x = self.head(x) + c
        else:
            x = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))

        return L2Wrap.apply(loss, x)
