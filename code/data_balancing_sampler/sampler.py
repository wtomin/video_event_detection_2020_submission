import os
import sys
from PIL import Image
import six
import string
import numpy as np
import lmdb
import pickle
import tqdm
import pyarrow as pa
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from collections import namedtuple
Segment = namedtuple('Segment', ['video', 'start', 'end', 'label'])
"""
key	value
seq-id1	(feature_seq1, label_seq1)
seq-id2	(feature_seq12, label_seq2)
seq-id3	(feature_seq3, label_seq3)
"""
def dumps_pyarrow(obj):
	"""
	Serialize an object.
	Returns:
		Implementation-dependent bytes-like object
	"""
	return pa.serialize(obj).to_buffer()
# data balancing is only applied to training set.
class balancing_sampler(object):
	def __init__(self, input_dir, output_path, json_file,
		target_num_seqs = 4732, target_length_seq = 20, 
		median_class_duration = 23659):
		"""
		input_dir: directory containing training .npy features
		output_path: lmdb file path to be saved
		json_file: training set annotations
		target_num_seqs: for each label, we sample # sequences
		target_length_seq: for each sequence, the length is # in seconds
		median_class_duration: the label with median duration, which is 23659 seconds
		"""
		self.input_dir = input_dir # input directory of the .npy features
		self.output_path = output_path # lmdb file path
		self.json_file = json_file
		# the median class is xiyan, the number of sequences to be samples is 4,732
		# the sequence length is 5s, 150 frames
		self.target_num_seqs = target_num_seqs
		self.target_length_seq = target_length_seq
		self.median_class_duration = median_class_duration # in seconds
		with open(self.json_file) as f:
			self.data = json.loads(f.readlines()[0])
		self.frame_rate = 15. # 15 frame per second 
		self.define_downsample_oversample_groups()

	def define_downsample_oversample_groups(self):
		data = self.data
		video_ids = data.keys()
		segment_label_durati = {}
		self.segment_list = []
		self.segment_dict = {}
		self.sid_2_same_vid_sids = []
		sid = 0
		for vid in tqdm(video_ids):
			segments = data[vid]
			annotations = segments['annotations']
			for annot in annotations:
				label = annot['label']
				if label not in segment_label_durati.keys():
					segment_label_durati[label] = 0
				dura = annot['segment'][1] - annot['segment'][0]
				assert dura >=0, "invalid segment!"
				segment_label_durati[label] += dura
				segment = Segment(video = vid, start = annot['segment'][0],
					end = annot['segment'][1], label = label)
				if label not in self.segment_dict.keys():
					self.segment_dict[label] = []
				self.segment_list.append(segment)
				self.segment_dict[label].append(sid)
				sid +=1
			len_segs = len(annotations)
			for i in range(len_segs):
				start = sid - len_segs
				list_ids = np.arange(len_segs).tolist()
				list_ids.remove(i)
				if list_ids is None or len(list_ids)==0:
					same_vid_sids = list()
				else:
					same_vid_sids =  [i+start for i in list_ids]
				self.sid_2_same_vid_sids.append(same_vid_sids)
		assert len(self.sid_2_same_vid_sids) == len(self.segment_list)
		d_group = []
		o_group = []
		self.label_list = sorted(segment_label_durati.keys())
		for label in self.label_list:
			if segment_label_durati[label] < self.median_class_duration:
				o_group.append(label)
			else:
				d_group.append(label)
		self.downsample_labels = d_group
		self.oversmaple_labels = o_group

	def sample_sequences(self, write_frequency=5000):
		print("Generate LMDB to %s" % self.output_path)
		isdir = os.path.isdir(self.output_path)
		db = lmdb.open(self.output_path, subdir=isdir,
					   map_size=1099511627776 * 2, readonly=False,
					   meminit=False, map_async=True)

		txn = db.begin(write=True)
		i_seq = 0
		for i_label, label in enumerate(self.label_list):
			segment_ids = self.segment_dict[label]
			same_video_segment_ids = [self.sid_2_same_vid_sids[i] for i in segment_ids]
			if label in self.downsample_labels:
				sample_weights = self.random_sample_weights(segment_ids, same_video_segment_ids)
				sampled_ids = np.random.choice(np.arange(len(segment_ids)), self.target_num_seqs, p = sample_weights)
				for i, id in enumerate(sampled_ids):
					sid = segment_ids[id]
					segment = self.segment_list[sid]
					video_segment_list = [self.segment_list[id] for id in same_sid]
					same_sid = same_video_segment_ids[id]
					sequence_id = "{}_{}".format(segment.video, sid)
					feature_seq, label_seq = self.sample_one_segment(segment, video_segment_list)
					txn.put(u'{}'.format(i_seq).encode('ascii'), dumps_pyarrow((feature_seq, label_seq )))
					i_seq +=1
					print("[{}/{}] Label {} sampling : [{}/{}]".format(i_label, len(self.label_list), label, i, self.target_num_seqs),end='\r')
					if i_seq % write_frequency == 0:
						txn.commit()
						txn = db.begin(write=True)
					del feature_seq, label_seq
			else:
				for i, (sid, same_sid) in enumerate(zip(segment_ids, same_video_segment_ids)):
					segment = self.segment_list[sid]
					video_segment_list = [self.segment_list[id] for id in same_sid]
					sequence_id = "{}_{}".format(segment.video, sid)
					feature_seq, label_seq = self.sample_one_segment(segment, video_segment_list)
					txn.put(u'{}'.format(i_seq).encode('ascii'), dumps_pyarrow((feature_seq, label_seq )))
					i_seq +=1
					if i_seq % write_frequency == 0:
						txn.commit()
						txn = db.begin(write=True)
					del feature_seq, label_seq
					print("[{}/{}] Label {} sampling : [{}/{}]".format(i_label, len(self.label_list), label, i, self.target_num_seqs),end='\r')
					if i == self.target_num_seqs - 1:
						break
				if i + 1 < self.target_num_seqs:
					N = self.target_num_seqs - (i+1)
					sample_weights = self.random_sample_weights(segment_ids, same_video_segment_ids)
					sampled_ids = np.random.choice(np.arange(len(segment_ids)), N, p = sample_weights)
					for id in sampled_ids:
						sid = segment_ids[id]
						same_sid = same_video_segment_ids[id]
						sequence_id = "{}_{}".format(segment.video, sid)
						feature_seq, label_seq = self.sample_one_segment(segment, video_segment_list)
						txn.put(u'{}'.format(i_seq).encode('ascii'), dumps_pyarrow((feature_seq, label_seq )))
						i_seq +=1
						if i_seq % write_frequency == 0:
							txn.commit()
							txn = db.begin(write=True)
						del feature_seq, label_seq
						i+=1
						print("[{}/{}] Label {} sampling : [{}/{}]".format(i_label, len(self.label_list), label, i, self.target_num_seqs),end='\r')
		print("All labels are sampled.")
		# finish iterating 
		txn.commit()
		keys = [u'{}'.format(k).encode('ascii') for k in range(i_seq + 1)]
		with db.begin(write=True) as txn:
			txn.put(b'__keys__', dumps_pyarrow(keys))
			txn.put(b'__len__', dumps_pyarrow(len(keys)))
		print("Database saved to %s"%self.output_path)
		db.sync()
		db.close()

	def random_sample_weights(self, segment_ids, same_video_segment_ids):
		weights = [1/float(len(ids)+1) for ids in same_video_segment_ids]
		return np.array(weights)/np.sum(np.array(weights))
	def downsample(self, segments):
		len_segs = len(segments)

	def sample_one_segment(self, input_segment, video_segment_list):
		video = input_segment.video
		label = input_segment.label
		npy_fpath = os.path.join(self.input_dir, '%s.npy'%video)
		target_length = int(self.target_length_seq * self.frame_rate) #10 * 15 = 150
		target_length = self.frame_length_to_feature_length(target_length) #floor(150/8)
		try:
			feature = np.load(npy_fpath)
		except:
			raise ValueError("{} does not exist!".format(npy_fpath))
		maximum_f_id = feature.shape[0]
		if maximum_f_id <= target_length:
			N = target_length - maximum_f_id
			if N == 0:
				feature_ids = np.arange(maximum_f_id)
			else:
				feature_ids = np.concatenate([np.arange(maximum_f_id), np.array([maximum_f_id-1]*N).astype(np.int)])
			feature_seq = feature[feature_ids,:]
			x = self.label_list.index(label)
			label_seq = np.eye(len(self.label_list))[np.ones(target_length).astype(np.int)*x]
		else:
			sampled_start, sampled_end = self.randomly_select_seq_range(input_segment.start, input_segment.end,
				maximum = maximum_f_id)

			if sampled_end - sampled_start + 1 != target_length:
				N = target_length - (sampled_end - sampled_start + 1)
				if N < 0:
					sampled_end = sampled_end + N
					feature_ids = np.arange(sampled_start, sampled_end+1)
				else:
					if sampled_end == maximum_f_id - 1:
						feature_ids = np.concatenate([np.arange(sampled_start, sampled_end+1), np.array([maximum_f_id - 1]*N).astype(np.int)])
					elif sampled_start == 0:
						feature_ids = np.concatenate([np.zeros(N).astype(np.int), np.arange(sampled_start, sampled_end+1)])
					else:
						feature_ids = np.concatenate([np.arange(sampled_start, sampled_end+1), np.array([sampled_end]*N).astype(np.int)])
			else:
				feature_ids = np.arange(sampled_start, sampled_end + 1)
			feature_seq = feature[feature_ids,:]
			x = self.label_list.index(label)
			label_seq = np.eye(len(self.label_list))[np.ones(target_length).astype(np.int)*x]
			out_of_range = feature_ids < self.second_2_id(input_segment.start)
			label_seq[out_of_range] = np.zeros_like(label_seq[out_of_range])
			out_of_range = feature_ids > self.second_2_id(input_segment.end)
			label_seq[out_of_range] = np.zeros_like(label_seq[out_of_range])
		label_seq = self.detect_mutiple_labels(label_seq, feature_ids, video_segment_list)
		label_seq = np.clip(label_seq, 0, 1)
		return feature_seq, label_seq 
	
	def detect_mutiple_labels(self, label_seq, input_feature_range, video_segment_list):
		if len(video_segment_list) == 0:
			return label_seq
		for segment in video_segment_list:
			start, end = segment.start, segment.end
			segment_feature_range = np.arange(self.second_2_id(start), self.second_2_id(end)+1)
			intersection, intersection_id, _ = np.intersect1d(input_feature_range, segment_feature_range, return_indices=True)
			if len(intersection) == 0:
				return label_seq
			else:
				N = len(intersection)
				new_label = self.label_list.index(segment.label)
				new_label = np.eye(len(self.label_list))[np.ones(N).astype(np.int)*new_label]
				label_seq[intersection_id] = label_seq[intersection_id] + new_label
				return label_seq

	def randomly_select_seq_range(self, start, end, maximum):
		target_length = int(self.target_length_seq * self.frame_rate) #10 * 15 = 150
		target_length = self.frame_length_to_feature_length(target_length) #floor(150/8)
		search_ids = np.arange(self.second_2_id(start), self.second_2_id(end)+1)
		sampled_middle = np.random.choice(search_ids, 1)[0]
		new_start = sampled_middle - target_length // 2
		new_end = sampled_middle + target_length // 2 - 1
		if new_start <0:
			new_start = 0
			new_end = new_start + target_length - 1
		if new_end > maximum - 1:
			new_end = maximum - 1
			new_start = new_end - target_length + 1
		return max(0, new_start), min(new_end, maximum-1)
		
	def second_2_id(self, input_time):
		frame_id = input_time*self.frame_rate
		feature_id = self.frame_length_to_feature_length(frame_id)
		return feature_id - 1
	def id_2_second(self, input_feature_id):
		input_frame_id_min, input_frame_id_max = self.feature_length_to_frame_length(input_feature_id+1)
		time = (input_frame_id_min+input_frame_id_max)*0.5/self.frame_rate
		return time
	def feature_length_to_frame_length(self, N):
		return int(N*8), int((N+1)*8)
	def frame_length_to_feature_length(self, T):
		return int(np.floor(T/8))

if __name__=='__main__':
	sampler = balancing_sampler(input_dir = '../../data/Train/i3d_features',
		output_path = '../../user_data/Train/i3d_features.lmdb',
		json_file = '../../data/Train/train_annotations.json')
	sampler.sample_sequences()

	save_label_list = "../../user_data/label_list.txt"
	with open(save_label_list, 'w') as out:
		out.write(",".join(sampler.label_list))
	



