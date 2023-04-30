import json
import polars as pl

from torch.utils.data import Dataset

class BaseDataset (Dataset):

	def __init__(self) -> None:

		super().__init__()

	def _parse_meta(self, meta) -> pl.DataFrame:

		# extracts files and labels into lists
		with open(meta, 'r') as fid:
			files, labels = zip(*json.load(fid).items())

		# places the data into a polars DataFrame
		return pl.DataFrame({
				'fname' : files,
				'label' : labels
		})

	def __len__(self):
		raise NotImplementedError()

	def __getitem__(self, index):
		raise NotImplementedError()