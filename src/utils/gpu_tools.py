from src.utils.globalVariable import GLOBAL_VARIABLE
import torch

def move2cuda(item):
    if isinstance(item, list) or isinstance(item, tuple):
        for id in range(len(item)):
            item[id] = move2cuda(item[id])
    else:
        if item is not None:
            if GLOBAL_VARIABLE.DEFAULT_GPU is not None:
                if torch.cuda.current_device() != 0:
                    item = item.cuda(torch.cuda.current_device())
                else:
                    item = item.cuda(GLOBAL_VARIABLE.DEFAULT_GPU)
            else:
                item = item.cuda()
    return item

def move2cpu(item):
    if isinstance(item, list) or isinstance(item, tuple):
        for id in range(len(item)):
            item[id] = move2cpu(item[id])
    else:
        if item is not None:
            item = item.cpu()
    return item
