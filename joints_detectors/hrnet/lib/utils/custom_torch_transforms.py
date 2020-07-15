import torch


class NormalizeEachImg:
    def __init__(self):
        pass

    def __call__(self, img):
        _mean, _std = torch.mean(img), torch.std(img)
        img = (img - _mean) / (_std + 1e-8)
        return img
