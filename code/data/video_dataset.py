# this video dataloader is used when the input data are the i3d features of one video file. No label is available
import torch
import numpy as np 
import os

class Video_Dataset(torch.utils.data.Dataset):
	def __init__(self, video_id, video_feature_path, seq_length, transform=None):
		self.video_id = video_id
		self.video_feature_path = video_feature_path
		self.video_feature = np.load(self.video_feature_path)
		self.seq_length = seq_length
		self.transform = transform
		self.sample_seqeuences()

	def sample_seqeuences(self):
		N = len(self.video_feature) // self.seq_length
		self.seq_list = []
		self.ids_list = []
		for i in range(N):
			self.seq_list.append(self.video_feature[i*self.seq_length:(i+1)*self.seq_length, :])
			self.ids_list.append([i*self.seq_length, (i+1)*self.seq_length-1])
		if N*self.seq_length < len(self.video_feature):
			self.seq_list.append(self.video_feature[N*self.seq_length:,:])
			self.ids_list.append([N*self.seq_length, len(self.video_feature)-1])
		self.length = len(self.seq_list)
	def __len__(self):
		return self.length
	def __getitem__(self, index):
		assert index<=self.length
		feature = self.seq_list[index]
		ids = self.ids_list[index]
		if self.transform is not None:
			feature = self.transform(feature)
		return {'index': index, 'image': feature, 'IDs': ids}





