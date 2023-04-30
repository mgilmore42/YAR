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
			convs = [30, 50, 100, 175, 250, 325, 375, 425, 500],
			activation=nn.Sigmoid,
			**kargs
		) -> None:

		super().__init__()

		self.weights = nn.Sequential()

		# inital block
		self.weights.append(
			nn.Sequential(
				Conv(3,convs[0],**kargs),
				activation(),
				ResidualBlock(
					convs[0],
					blocks=6,
					activation=activation,
					**kargs
				),
				nn.MaxPool2d(2),
				activation()
			)
		)

		# remiaining blocks
		for in_s, out_s in zip(convs[:-1],convs[1:]):
			self.weights.append(
				nn.Sequential(
					Conv(in_s, out_s,**kargs),
					nn.MaxPool2d(2),
					activation(),
					ResidualBlock(
						out_s,
						blocks=2,
						activation=activation,
						**kargs
					),
					nn.BatchNorm2d(out_s, dtype=kargs.get('dtype'))
				)
			)

	def forward(self,data):
		return self.weights(data)

class Decoder(nn.Module):

	def __init__(
			self,
			convs = [30, 50, 100, 175, 250, 325, 375, 425, 500],
			activation=nn.Sigmoid,
			**kargs
		) -> None:

		super().__init__()

		self.weights = nn.Sequential()

		convs = convs[::-1]

		# middle blocks
		for in_s, out_s in zip(convs[:-1],convs[1:]):

			self.weights.append(
				nn.Sequential(
					ConvTrans(  in_s, **kargs),
					Conv(in_s, out_s, **kargs),
					activation(),
					ResidualBlock(
						out_s,
						blocks=2,
						activation=activation,
						**kargs
					),
					nn.BatchNorm2d(out_s, dtype=kargs.get('dtype'))
				)
			)

		# end block
		self.weights.append(
			nn.Sequential(
				ConvTrans(convs[-1],**kargs),
				ResidualBlock(
					convs[-1],
					blocks=6,
					activation=activation,
					**kargs
				),
				activation(),
				Conv(convs[-1], 3,**kargs),
				activation()
			)
		)

	def forward(self,data):
		return self.weights(data)

class Conv(nn.Module):

	def __init__(
			self,
			in_channel,
			out_channel,
			initialization=nn.init.xavier_uniform_,
			depthwise=False,
			dtype=torch.float,
			**_
		) -> None:

		super().__init__()

		if depthwise is False:
			self.conv = nn.Conv2d(in_channel, out_channel, 3, padding= 1, dtype=dtype)

			initialization(self.conv.weight)
		else:
			depthwise = nn.Conv2d(in_channel,  in_channel, 3, padding=1, groups=in_channel,dtype=dtype)
			pointwise = nn.Conv2d(in_channel, out_channel, 1, dtype=dtype)

			initialization(depthwise.weight)
			initialization(pointwise.weight)

			self.conv = nn.Sequential(
				depthwise,
				pointwise
			)

	def forward(self, data):
		return self.conv(data)

class ConvTrans(nn.Module):

	def __init__(
			self,
			in_channel,
			initialization=nn.init.xavier_uniform_,
			trans_depthwise=False,
			dtype=torch.float,
			**_
		) -> None:

		super().__init__()

		if trans_depthwise is False:
			self.conv_trans = nn.ConvTranspose2d(in_channel,in_channel, 3, padding=1, stride=2, output_padding=1,dtype=dtype)
		else:
			self.conv_trans = nn.ConvTranspose2d(in_channel,in_channel, 3, padding=1, stride=2, output_padding=1, groups=in_channel,dtype=dtype)

		initialization(self.conv_trans.weight)

	def forward(self, data):
		return self.conv_trans(data)

class ResidualBlock(nn.Module):

	def __init__(
			self,
			size,
			blocks=4,
			activation=nn.Sigmoid,
			**kargs
		) -> None:

		super().__init__()

		self.weights = nn.Sequential()

		for _ in range(blocks):

			self.weights.append(
				nn.Sequential(
					Conv(size, size, **kargs),
					activation(),
				)
			)

	def forward(self, data):
		return data + self.weights(data)

class ResidualConnection(nn.Module):

	def __init__(self, *blocks) -> None:
		super().__init__()

		self.model = nn.Sequential(
			*blocks
		)

	def forward(self, data):
		return data + self.model(data)