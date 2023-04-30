#!/usr/bin/python

from models.autoender import Autoencoder

from utils.mask       import mask
from utils.data       import AEDataset
from utils.transforms import Normalize

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

logger = SummaryWriter()

dataset_origin = torch.utils.data.ConcatDataset([AEDataset(day=1), AEDataset(day=2)])

dataset = dataset_origin
val_dataset_roi = AEDataset(day=2)
dataset, val_dataset, *_ = torch.utils.data.random_split(dataset_origin,[0.5,0.5])
dataset, *_ = torch.utils.data.random_split(dataset_origin,[0.5,0.5])
dataset, *_ = torch.utils.data.random_split(dataset_origin,[0.5,0.5])

data_transforms=nn.Sequential(
	Normalize(),
	transforms.Resize((512,512),antialias=True),
	Normalize()
)

test_loader = DataLoader(dataset,num_workers=8,shuffle=True,batch_size=35)
val_loader = DataLoader(val_dataset,num_workers=8,shuffle=True,batch_size=100)

criterion = nn.MSELoss(reduction='mean')

loss_indecies = mask(512,255).cuda()

loss_mask = loss_indecies.bfloat16().unsqueeze(0).unsqueeze(0)

output_normalize = Normalize()

model = Autoencoder(
	convs = [30, 40, 70, 120, 180, 260, 360, 480, 600],
	activation=nn.LeakyReLU,
	initialization=nn.init.kaiming_normal_,
	depthwise=False,
	trans_depthwise=False,
	dtype=torch.bfloat16
).cuda()

optimizer = torch.optim.SGD(model.parameters(),lr=1,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

best_loss = float('inf')

for epoch in range(60):

	running_loss = 0.0

	model.train()

	for i, img in enumerate(test_loader):

		optimizer.zero_grad(set_to_none=True)

		img = data_transforms(img.cuda().float()).bfloat16() * loss_mask

		out = output_normalize(model(img)) * loss_mask

		loss = criterion(out[:,:,loss_indecies], img[:,:,loss_indecies])

		if torch.isnan(loss):
			print(f'batch {i} NaN loss')
		else:
			loss.backward()

		optimizer.step()

		running_loss += img.shape[0] * loss.item()

		del loss, out, img

	epoch_loss = running_loss/len(dataset)

	# logs epoch training loss
	logger.add_scalar('AE MSE Trian loss', epoch_loss, epoch)
	logger.flush()

	model.eval()

	with torch.no_grad():

		data = torch.stack((
			torch.zeros((3,512,512)).float(),
			torch.ones((3,512,512)).float(),
			0.5*torch.ones((3,512,512)).float(),
			data_transforms(val_dataset_roi[-1].float()),
			data_transforms(val_dataset_roi[0].float()),
			data_transforms(val_dataset_roi[7780].float()),
			data_transforms(val_dataset_roi[9418].float())
		)).cuda()

		predictions = output_normalize(model(data.bfloat16()))
		masked_preds = predictions * loss_mask

		data         = (255*data        ).type(torch.uint8).permute(1,0,2,3).reshape((3,-1,512)).cpu()
		predictions  = (255*predictions ).type(torch.uint8).permute(1,0,2,3).reshape((3,-1,512)).cpu()
		masked_preds = (255*masked_preds).type(torch.uint8).permute(1,0,2,3).reshape((3,-1,512)).cpu()

		img = torch.stack((data,masked_preds,predictions),dim=2).reshape(3,512*7,-1)

		torchvision.io.write_png(img,filename=f'.cache/res25/{epoch}.png',compression_level=9)

		logger.add_images('ROIs', torch.stack((data,masked_preds,predictions)),global_step=epoch)
		logger.flush()

		del predictions, masked_preds

		for img in val_loader:

			img = data_transforms(img.cuda().float()).bfloat16() * loss_mask

			out = output_normalize(model(img)) * loss_mask

			loss = criterion(out[:,:,loss_indecies], img[:,:,loss_indecies]).item()

			running_loss += img.shape[0] * loss

			del loss, out, img

		logger.add_scalar('AE MSE Validation Loss', running_loss/len(val_dataset),epoch)
		logger.flush()


		if best_loss > running_loss:
			best_loss = running_loss
			torch.save(model.state_dict(), '.cache/check_points/model 25.pt')

	if (epoch % 10) == 5:
		scheduler.step()

logger.close()