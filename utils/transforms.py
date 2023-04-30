import torch
import torch.nn as nn

class Normalize(nn.Module):

	def __init__(self) -> None:
		super().__init__()

	def forward(self, data : torch.Tensor):
		data_view = data.view((*data.shape[:-2],1,-1))
		min, _ = data_view.min(dim=-1, keepdim=True)
		max, _ = data_view.max(dim=-1, keepdim=True)

		return (data - min) / (max - min)