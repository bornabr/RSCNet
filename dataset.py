import glob
import torch
import numpy as np


def UT_HAR_dataset(root_dir):
	"""Reads UT_HAR dataset and returns WiFi data as tensors.

	Args:
		root_dir (string): Root directory containing UT_HAR data and label files.

	Returns:
		dict: Dictionary containing WiFi data as tensors.
	"""
	data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
	label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
	WiFi_data = {}

	# Process data files
	for data_dir in data_list:
		data_name = data_dir.split('/')[-1].split('.')[0]
		with open(data_dir, 'rb') as f:
			data = np.load(f)
			data = data.reshape(len(data),1,250,90)
			data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
		WiFi_data[data_name] = torch.Tensor(data_norm)
	
	 # Process label files
	for label_dir in label_list:
		label_name = label_dir.split('/')[-1].split('.')[0]
		with open(label_dir, 'rb') as f:
			label = np.load(f)
		WiFi_data[label_name] = torch.Tensor(label).to(torch.int64)

	# Shape 1 x 250 (Time) x 90 (antenna x subcarrier)
	return WiFi_data

def create_loader_from_dataset(train_set, val_set, test_set, batch_size, num_workers):
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)
	if val_set:
		val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		return train_loader, val_loader, test_loader
	else:
		return train_loader, None, test_loader

def load_UT_HAR_dataset(root, batch_size, num_workers):
	data = UT_HAR_dataset(root)
	train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
	val_set = torch.utils.data.TensorDataset(data['X_val'], data['y_val'])
	test_set = torch.utils.data.TensorDataset(data['X_test'], data['y_test'])
	
	return create_loader_from_dataset(train_set, val_set, test_set, batch_size, num_workers)

def data_loader(cfg, validation_split=0.2, num_workers=20):
	root = cfg['root_dir']
	batch_size = cfg['batch_size']

	if cfg['name'] == 'UT_HAR':
		return load_UT_HAR_dataset(root, batch_size, num_workers)
	else:
		raise 'Unknown dataset'
