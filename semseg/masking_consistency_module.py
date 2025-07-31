import random

import torch
from torch.nn import Module

from utils.utils import BlockMaskGenerator


class MaskingConsistencyModule(Module):

    def __init__(self, mask_ratio=0.7, mask_block_size=64):
        super(MaskingConsistencyModule, self).__init__()
        self.mask_gen = BlockMaskGenerator(mask_ratio=mask_ratio, mask_block_size=mask_block_size)

