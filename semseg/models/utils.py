import os
import gzip
import html
import torch
import torch.nn.functional as F

import random
import ftfy
import regex as re
import numpy as np
import itertools

from functools import lru_cache
from typing import Any, Union, List

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


_tokenizer = SimpleTokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    # print(result.shape)
    # exit()

    return result

def get_text_embeddings(texts, text_encoder): # list of texts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # texts = add_synonym(texts)
    text_tokens = torch.cat([tokenize(
                            texts = f"a clean origami of a {text}",
                            context_length = text_encoder.context_length) # = 13
                        for text in texts]).to(device)

    # (T, C), where T= # of templetes, C=512
    text_embeddings = text_encoder(text_tokens)
    return text_embeddings  

def add_synonym(texts):
    for i, text in enumerate(texts):
        if text == "person":
            texts[i] = "person, standing person, person with clothes"
        if text == "terrain":
            texts[i] = "terrain, short plant, grass on the ground"
        if text == "rider":
            texts[i] = "rider, rider on the bicycle, rider on the bicycle"
        if text == "train":
            texts[i] = "train, railroad car"
    return texts

def Sinkhorn_log_exp_sum(C, mu, nu, epsilon):
    
    def _log_boltzmann_kernel(u, v, epsilon, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= epsilon
        return kernel
  
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-6
    max_iter = 100
            
    for i in range(max_iter):
       
        u0 = u  # useful to check the update
        K = _log_boltzmann_kernel(u, v, epsilon, C)
        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
        u = epsilon * u_ + u
        
        K_t = _log_boltzmann_kernel(u, v, epsilon, C).permute(0, 2, 1).contiguous()
        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
        v = epsilon * v_ + v
        
        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break
    
    K = _log_boltzmann_kernel(u, v, epsilon, C)
    T = torch.exp(K)

    return T

def mask_modalities(A, mask_value=0.0):
    """
    A: Tensor of shape [m, b, c, h, w]
    mask_value: the value to set for masked modalities (default: 0.0)
    """
    m = A.shape[0]
    
    num_mask = random.randint(0, m-1)  
    
    mask_indices = random.sample(range(m), num_mask)

    mask = torch.ones_like(A)
    for idx in mask_indices:
        mask[idx] = 0.0

    A_masked = A * mask + mask_value * (1 - mask)
    
    return A_masked, mask_indices


def sample_modality_mask(m, b):
    all_combos = []
    for r in range(1, m + 1):
        all_combos.extend(itertools.combinations(range(m), r))

    all_combos = np.array([
        np.isin(range(m), combo).astype(np.float32)
        for combo in all_combos
    ])  # shape: [num_combos, m]

    sampled = all_combos[np.random.choice(len(all_combos), size=b, replace=True)]  # [b, m]

    return torch.tensor(sampled.T, dtype=torch.float32)  # [m, b]



def sample_modality_mask_v2(m, b):
    all_combos = []
    for r in range(1, m + 1):
        all_combos.extend(itertools.combinations(range(m), r))

    selected = all_combos[np.random.choice(len(all_combos))]  

    mask = np.isin(range(m), selected).astype(np.float32)  # [m]

    return torch.tensor(mask, dtype=torch.long).unsqueeze(-1).expand(-1, b)  # [m]


def gradient_conflict_minimization(shared_param, specific_param):
    """
    """
    g_s = shared_param.grad  # [D]
    g_sp = specific_param.grad  # [D]

    if g_s is None or g_sp is None:
        return

    # flatten
    g_s_flat = g_s.view(-1)
    g_sp_flat = g_sp.view(-1)

    g_sp_norm = torch.norm(g_sp_flat) + 1e-8
    unit_g_sp = g_sp_flat / g_sp_norm

    proj = torch.dot(g_s_flat, unit_g_sp) * unit_g_sp  # [D]

    g_s_orth = g_s_flat - proj

    shared_param.grad.copy_(g_s_orth.view_as(shared_param))
