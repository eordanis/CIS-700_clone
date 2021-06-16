import time
from time import strftime, localtime

import os
import re
import torch

k_label = 2  # num of labels, >=2
mem_slots = 1  # RelGAN-1
num_heads = 2  # RelGAN-2
head_size = 256  # RelGAN-256
start_letter = 1
padding_idx = 0