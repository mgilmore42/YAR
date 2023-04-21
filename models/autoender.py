import torch
import torch.nn as nn


class Autoencoder(nn.Module):

	def __init__(self, **kargs) -> None:

		super().__init__()

		self.encoder = Encoder(**kargs)
		self.decoder = Decoder(**kargs)

	def forward(self,data):
		return self.decoder(
			self.encoder(data)
		)

class Encoder(nn.Module):

	def __init__(
			self,
			convs = None,
			activation=nn.Sigmoid,
			initialization=nn.init.xavier_uniform_
		) -> None:

		super().__init__()

		if convs is None:
			convs = [3, 10, 20, 30, 40, 50, 60, 70, 85, 100]

		assert len(convs) == 10

		self.weights = nn.Sequential()

		for in_s, out_s in zip(convs[:-1],convs[1:]):

			# allocates conv layer
			conv = nn.Conv2d(in_s, out_s, 3, padding=1)

			# initializes the conv layer
			initialization(conv.weight)

			block = nn.Sequential(
				conv,
				nn.MaxPool2d(2),
				activation(),
				nn.BatchNorm2d(out_s),
				ResidualConnection(
					nn.Sequential(
						ResidualBlock(out_s),
						ResidualBlock(out_s)
					)
				)
			)

			self.weights.append(block)

	def forward(self,data):
		return self.weights(data)

class Decoder(nn.Module):

	def __init__(
			self,
			convs = None,
			activation=nn.Sigmoid,
			initialization=nn.init.xavier_uniform_
		) -> None:

		super().__init__()

		if convs is None:
			convs = [3, 10, 20, 30, 40, 50, 60, 70, 85, 100][::-1]

		# assert len(convs) == 9

		self.weights = nn.Sequential()

		for in_s, out_s in zip(convs[:-1],convs[1:]):

			# allocates conv layer
			conv = nn.ConvTranspose2d(in_s, out_s, 3, padding=1, stride=2, output_padding=1)

			# initializes the conv layer
			initialization(conv.weight)

			block = nn.Sequential(
				conv,
				activation(),
				nn.BatchNorm2d(out_s),
				ResidualConnection(
					nn.Sequential(
						ResidualBlock(out_s),
						ResidualBlock(out_s)
					)
				)
			)

			self.weights.append(block)

	def forward(self,data):
		return self.weights(data)

class ResidualBlock(nn.Module):

	def __init__(
			self,
			size,
			blocks=3,
			activation=nn.Sigmoid,
			initialization=nn.init.xavier_uniform_
		) -> None:

		super().__init__()

		self.weights = nn.Sequential()

		for block in range(blocks):
			# allocates conv layer
			conv = nn.Conv2d(size, size, 3, padding=1)

			# initializes the conv layer
			initialization(conv.weight)

			block = nn.Sequential(
				conv,
				activation(),
			)

			self.weights.append(block)

	def forward(self, data):
		return data + self.weights(data)

class ResidualConnection(nn.Module):

	def __init__(self, model) -> None:
		super().__init__()

		self.model = model

	def forward(self, data):
		return data + self.model(data)