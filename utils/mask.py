import torch

def mask(size, radius):

	basis = (torch.arange(size) - (size -1)/2)**2

	mask = (basis.unsqueeze(1) + basis.unsqueeze(0))**0.5

	return mask < radius