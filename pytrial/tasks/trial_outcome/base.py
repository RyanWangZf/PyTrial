import abc
import os

from torch.utils.data import DataLoader

class TrialOutcomeBase(abc.ABC):
	'''Abstract class for all clinical trial outcome prediction

	'''
	@abc.abstractmethod
	def __init__(self):
		pass

	@abc.abstractmethod
	def fit(self, train_data, valid_data):
		raise NotImplementedError

	@abc.abstractmethod
	def predict(self, test_data):
		raise NotImplementedError

	@abc.abstractmethod
	def load_model(self, checkpoint):
		raise NotImplementedError

	@abc.abstractmethod
	def save_model(self, checkpoint):
		raise NotImplementedError
	
	def _build_dataloader_from_dataset(
		self,
		dataset,
		batch_size=64,
		shuffle=False,
		num_workers=0,
		collate_fn=None,
		):
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
		return dataloader

