from models.autoender import Autoencoder

from utils.dataloader import PhoDS
from utils.mask import mask

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
# import torch.functional as F

import os

log = SummaryWriter()

dataset = PhoDS()

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad) 

class Normalize(nn.Module):

	def __init__(self) -> None:
		super().__init__()

	def forward(self, data : torch.Tensor):
		data_view = data.view((*data.shape[:-2],1,-1))
		min, _ = data_view.min(dim=-1, keepdim=True)
		max, _ = data_view.max(dim=-1, keepdim=True)

		return (data - min) / (max - min)

data_transforms=nn.Sequential(
	Normalize(),
	transforms.Resize((512,512),antialias=True),
)

train_transforms=nn.Sequential(
	transforms.RandomRotation((0,360))
)

test_loader = DataLoader(dataset,num_workers=1,shuffle=True,batch_size=35)

criterion = nn.MSELoss(reduction='mean')

loss_indecies = mask(512,255).cuda()

loss_mask = loss_indecies.float().unsqueeze(0).unsqueeze(0)

output_normalize = Normalize()

model = Autoencoder(
	convs = [30, 40, 70, 120, 180, 260, 360, 480, 600],
	activation=nn.LeakyReLU,
	initialization=nn.init.kaiming_normal_,
	depthwise=False,
	trans_depthwise=False
).cuda()

optimizer = torch.optim.SGD(model.parameters(),lr=1,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

print(f'number of parameters {count_parameters(model)}')

best_loss = float('inf')

for epoch in range(200):

	running_loss = 0.0

	model.train()

	for i, (img, _) in enumerate(test_loader):

		optimizer.zero_grad(set_to_none=True)

		img = data_transforms(img.cuda().float())

		with torch.no_grad():
			train_img = (train_transforms(img)  * loss_mask).bfloat16()

		img = img.bfloat16() * loss_mask

		with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
			out = output_normalize(model(img)) * loss_mask

			loss = criterion(out[:,:,loss_indecies], img[:,:,loss_indecies])

			if torch.isnan(loss):
				print(f'batch {i} NaN loss')
			else:
				loss.backward()

			optimizer.step()

			scale_factor, *_ = img.shape

			running_loss += scale_factor * loss.item()

		log_img = img[-1,:].detach().float().cpu()
		log_out = out[-1,:].detach().float().cpu()

		del loss, out, img, train_img

	epoch_loss = running_loss/len(dataset)

	model.eval()

	with torch.no_grad():
		labels = ['zeros', 'ones', 'halfs', 'low illumination', 'high illumination empty', 'high illumination full', 'high illumination mid']

		data = torch.stack((
			torch.zeros((3,512,512)).float(),
			torch.ones((3,512,512)).float(),
			0.5*torch.ones((3,512,512)).float(),
			data_transforms(dataset[-1][0].float()),
			data_transforms(dataset[0][0].float()),
			data_transforms(dataset[7780][0].float()),
			data_transforms(dataset[6356][0].float())
		)).cuda()

		predictions = output_normalize(model(data))
		masked_preds = predictions * loss_mask.float()

		data         = (255*data        ).type(torch.uint8).permute(1,0,2,3).reshape((3,-1,512)).cpu()
		predictions  = (255*predictions ).type(torch.uint8).permute(1,0,2,3).reshape((3,-1,512)).cpu()
		masked_preds = (255*masked_preds).type(torch.uint8).permute(1,0,2,3).reshape((3,-1,512)).cpu()

		img = torch.stack((data,masked_preds,predictions),dim=2).reshape(3,512*7,-1)

		torchvision.io.write_png(img,filename=f'.cache/res/{epoch}.png',compression_level=9)

		log.add_images('ROIs', torch.stack((data,masked_preds,predictions)),global_step=epoch)
		log.flush()

	if (epoch % 10) == 5:
		scheduler.step()

	log.add_scalar('loss', epoch_loss, epoch)

torch.save(model.state_dict(), '.cache/check_points/model.pt')

log.close()