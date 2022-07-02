import numpy as np
import torch

valid_anchor_boxes = torch.arange(12).reshape(-1, 4)

for num1, i in enumerate(valid_anchor_boxes):
    ya1, xa1, ya2, xa2 = i
    print(ya1)
    print(xa1)
    print(ya2)
    print(xa2)
    print(num1, " ", i)