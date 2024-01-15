import json
import torch
import numpy as np
from torch.nn import functional as F

import ncompass.internal.logging as nclog
from ncompass.internal.models import NCTokenizer

# === Imported code from ridgerchu/SpikeGPT (src/utils.py)
class SpikeGPTTokenizer(NCTokenizer):
    def __init__(self, word_name, unknown_char='\ue083'):
        if 'list' in str(type(word_name)):
            self.charMode = False
            if word_name[0] == word_name[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=word_name[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(word_name[0], word_name[1])
            self.vocab_size = len(self.tokenizer)
        else:
            self.charMode = True
            with open(word_name + '.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.unknown_char = self.stoi[unknown_char]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        lastChar = int(x[-1])

        probs = F.softmax(torch.tensor(out), dim=-1)

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
        else:
            top_p = top_p_usual

        sorted_probs, s_index = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

        probs[probs < cutoff] = 0

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return torch.multinomial(probs, num_samples=1)[0]

def tokenizer_check(tokenizer: NCTokenizer, mode: str) -> None:
    if mode == 'pile':
        if (tokenizer.tokenizer.decode([187]) != '\n'):
            nclog.ERROR("Pile tokenizer does not decode 187 to new-line character!",
                        ValueError)
# =======================================================

# === Newly written code ================================
