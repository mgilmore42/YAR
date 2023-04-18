import json
import polars as pl
import torchvision.io as io

from torch.utils.data import Dataset

class PhoDS (Dataset):

	def __init__(
			self,
			tranforms = None,
			dir='data/PHO_117/Day_1_Oct_12/Camera 1',
			meta='data/PHO_117/Day_1_Oct_12/gt_day_1.json'
		) -> None:

		super().__init__()

		self.dir = dir

		self._parse_meta(meta)

		# sets data
		if tranforms is None:
			self.getitem = self._getitem
		else:
			self.transfroms = tranforms
			self.getitem = self._getitem_with_transforms

	def _parse_meta(self, meta):

		# extracts files and labels into lists
		with open(meta, 'r') as fid:
			files, labels = zip(*json.load(fid).items())

		# places the data into a polars DataFrame
		self.meta = pl.DataFrame({
				'fname' : files,
				'label' : labels
		})

	def __len__(self):
		return len(self.meta)

	def __getitem__(self, index):
		file, label = self.meta[index]

		label = label.item()

		img = io.read_image(f'{self.dir}/{file.item()}.jpg')

		if self.transfroms is None:
			return img, label
		else:
			return self.transfroms(img), label

	def _getitem(self, index):
		file, label = self.meta[index]

		img = io.read_image(f'{self.dir}/{file.item()}.jpg')

		return img, label.item()

	def _getitem_with_transforms(self, index):
		img, label = self._getitem(index)

		return self.transfroms(img), label