import torchvision.io as io
import os.path as path

from .dataset import BaseDataset


class AEDataset (BaseDataset):

	def __init__(
			self,
			day : int = 1,
		) -> None:

		super().__init__()

		if day == 1:
			self.img_dir = 'data/PHO_117/Day_1_Oct_12/Camera 1'
			self.meta    = self._parse_meta('data/PHO_117/Day_1_Oct_12/gt_day_1.json')
		elif day == 2:
			self.img_dir = 'data/PHO_117/Day_2_Oct_13/Camera 1'
			self.meta    = self._parse_meta('data/PHO_117/Day_2_Oct_13/gt_day_2.json')
		else:
			raise KeyError(f'There is no day for day {day}')

	def __len__(self):
		return len(self.meta)

	def __getitem__(self, index):
		file, _ = self.meta[index]

		img = io.read_image(path.join(self.img_dir, f'{file.item()}.jpg'))

		return img