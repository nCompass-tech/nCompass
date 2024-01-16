import gc
import types

import torch
import torch.nn as nn
from torch.nn import functional as F

#TODO: The following two libraries should also be rewritten as they are directly taken from 
# ridgerchu/SpikeGPT src/spikingjelly
from ncompass.internal.third_party.spikingjelly.clock_driven import neuron
from ncompass.internal.third_party.spikingjelly.clock_driven.surrogate import ATan

#TODO: Rewrite this if necessary, has been taken directly from ridgerchu/SpikeGPT
# src/model_run.py (class RWKV_RNN)
class SpikeGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.float_mode = config.float_mode
        self.device_type = config.device_type
        with torch.no_grad():
            w = torch.load(config.name_or_path, map_location='cpu')
            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(config.ctx_len+1, -1)[:-1,:]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // config.rwkv_rescale_layer))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // config.rwkv_rescale_layer))
                                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    if self.float_mode == "fp32":
                        w[x] = w[x].float()
                    elif self.float_mode == "bf16":
                        w[x] = w[x].bfloat16()
                    elif self.float_mode == "fp16":
                        w[x] = w[x].half()

                w[x].requires_grad = False
                if config.device_type == 'cuda' and x != 'emb.weight':
                    w[x] = w[x].cuda()

                if ('blocks.' not in x) or ('blocks.0.' in x):
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.config.n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw, mem):
        if self.float_mode == "bf16":
            xk = x * time_mix_k + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+0] = x.float()
        elif self.float_mode == "fp16":
            xk = x * time_mix_k + state[5*i+0].half() * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].half() * (1 - time_mix_r)
            state[5*i+0] = x.float()            
        else:
            xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
            state[5*i+0] = x

        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k

        return mem[i](r * kv)

    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, mem):
        if self.float_mode == "bf16":
            xk = x * time_mix_k + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+1] = x.float()
        elif self.float_mode == "fp16":
            xk = x * time_mix_k + state[5*i+1].half() * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1].half() * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1].half() * (1 - time_mix_r)
            state[5*i+1] = x.float()            
        else:
            xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
            state[5*i+1] = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        if '16' in self.float_mode:
            kk = k.float()
            vv = v.float()
        else:
            kk = k
            vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        if self.float_mode == "bf16":
            wkv = (a / b).type(torch.bfloat16)
        elif self.float_mode == "fp16":
            wkv = (a / b).half()
        else:
            wkv = a / b
        
        return mem[i](ow @ (r * wkv))

    def forward(self, ctx, state, mem1, mem2, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            config = self.config

            if self.config.vocab_size == 77:
                atan = ATan()
                x = atan(w.emb.weight[ctx[-1]])
            else:
                x = w.emb.weight[ctx[-1]]
            if self.device_type == 'cuda':
                x = x.cuda()
            try:
                pos_emb = w.pos_emb[len(ctx)-1]
                x = x + pos_emb
            except:
                pass
            
            if state == None:
                state = torch.zeros(config.num_hidden_layers * 5, config.n_embd, device=self.device_type)
                mem1 = []
                mem2 = []
                
                for i in range(config.num_hidden_layers):
                    state[5*i+4] -= 1e30
                    mem1.append(neuron.LIFNode())
                    mem2.append(neuron.LIFNode())
                    
            for i in range(config.num_hidden_layers):
                if i == 0:
                    x = self.LN(x, w.blocks[i].ln0)
                
                ww = w.blocks[i].att
                att = self.SA(self.LN(x, w.blocks[i].ln1), state, i,
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay,
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight, mem1)
                x = x + att
                ww = w.blocks[i].ffn
                ffn = self.FF(self.LN(x, w.blocks[i].ln2), state, i,
                    ww.time_mix_k, ww.time_mix_r,
                    ww.key.weight, ww.value.weight, ww.receptance.weight, mem2)

                x = x + ffn
                if (i+1) % config.rwkv_rescale_layer == 0:
                    x = x / 2

            if preprocess_only:
                return state, mem1, mem2

            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x
            
            return x.float(), state, mem1, mem2
