import os
import numpy
import math
import cv2

from skimage import io
from skimage import color

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from random import randint

class ClusterDataset(Dataset):
	"""Camera localization dataset."""



	def __cluster__(self, root_dir, num_experts):
		'''
		Clusters the dataset using hierarchical kMeans. 

		Initialization: 
			Put all images in one cluster.
		Interate: 
			Pick largest cluster.
			Split with kMeans and k=2.
			Input for kMeans is the 3D median scene coordiante per image.
		Terminate:
			When number of target clusters has been reached.

		Returns:
			cam_centers: For each cluster the mean (not median) scene coordinate
			labels: For each image the cluster ID
		'''

		print('Clustering the dataset ...')

		# load scene coordinate ground truth
		init_dir =  root_dir + '/init/'

		init_files = os.listdir(init_dir)
		init_files = [init_dir + f for f in init_files]
		init_files.sort()

		cam_centers = torch.zeros((len(init_files), 3))
		num_cam_clusters = num_experts

		# Calculate median scene coordinate per image
		for i, f in enumerate(init_files):

			init_data = torch.load(f)
			init_data = init_data.view(3, -1)

			init_mask = init_data.sum(0) != 0

			init_data = init_data[:,init_mask]
			init_data = init_data.median(1)[0]

			cam_centers[i] = init_data

		cam_centers = cam_centers.numpy()

		# setup kMEans
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
		flags = cv2.KMEANS_PP_CENTERS

		 # label of next cluster
		label_counter = 0

		# initialize list of clusters with all images
		clusters = []
		clusters.append((cam_centers, label_counter, numpy.zeros(3)))

		 # all images belong to cluster 0
		labels = numpy.zeros(len(init_files))

		# iterate kMeans with k=2
		while len(clusters) < num_cam_clusters:

			# select largest cluster (list is sorted)
			cur_cluster = clusters.pop(0)
			label_counter += 1

			# split cluster
			_, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

			# update cluster list
			cur_mask = (cur_labels == 0)[:,0]
			cur_cam_centers0 = cur_cluster[0][cur_mask,:]
			clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

			cur_mask = (cur_labels == 1)[:,0]
			cur_cam_centers1 = cur_cluster[0][cur_mask,:]
			clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

			cluster_labels = labels[labels == cur_cluster[1]]
			cluster_labels[cur_mask] = label_counter
			labels[labels == cur_cluster[1]] = cluster_labels

			#sort updated list
			clusters = sorted(clusters, key = lambda cluster: cluster[0].shape[0], reverse = True)

		cam_centers = torch.zeros(num_cam_clusters, 3)
		cam_sizes = torch.zeros(num_cam_clusters, 1)

		# output result
		for cluster in clusters:
					
			# recalculate cluster centers (is: mean of medians, we want: mean of means)
			
			cam_num = cluster[0].shape[0]
			cam_data = torch.zeros((cam_num, 3))
			cam_count = 0

			for i, f in enumerate(init_files):
				if labels[i] == cluster[1]:
					init_data = torch.load(f)
					init_data = init_data.view(3, -1)

					init_mask = init_data.sum(0) != 0

					init_data = init_data[:,init_mask]
					cam_data[cam_count] = init_data.sum(1) / init_mask.sum()
					cam_count += 1

			cam_centers[cluster[1]] = cam_data.mean(0)

			cam_dists = cam_centers[cluster[1]].unsqueeze(0).expand(cam_num, 3)
			cam_dists = cam_data - cam_dists
			cam_dists = cam_dists.norm(dim=1)
			cam_dists = cam_dists**2

			cam_sizes[cluster[1]] = cam_dists.mean()
		
			print("Cluster %i: %.1fm, %.1fm, %.1fm, images: %i" % (cluster[1], cam_centers[cluster[1]][0], cam_centers[cluster[1]][1], cam_centers[cluster[1]][2], cluster[0].shape[0]))
		
		print('Done.')
	
		return cam_centers, cam_sizes, labels




	def __init__(self, root_dir, num_clusters, cluster = -1, training=True, imsize=480, softness=5):
		'''Constructor.

		Clusters the dataset, and creates lists of data files.

		Parameters:
			root_dir: Folder of the data (training or test).
			expert: Select a specific cluster. Loads only data of this expert.
			num_experts: Number of clusters for clustering.
			training: Load and return ground truth scene coordinates (disabled for test).
		'''

		self.num_experts = num_clusters
		self.cluster = cluster
		self.training = training
		self.imsize = imsize		
		self.softness = softness

		with open('env_list.txt', 'r') as f:
			environment = f.readlines()

		if len(environment) > 1:
			print("ERROR: Environment file contains more than one line. Clustering is not supported for more than one dataset.")
			exit()

		root_dir = environment[0].strip() + '/' + root_dir

		rgb_dir = root_dir + '/rgb/'
		pose_dir =  root_dir + '/poses/'
		calibration_dir = root_dir + '/calibration/'
		init_dir =  root_dir + '/init/'
		seg_dir = root_dir + '/seg/'	

		self.rgb_files = os.listdir(rgb_dir)
		self.rgb_files = [rgb_dir + f for f in self.rgb_files]
		self.rgb_files.sort()

		# statistics calculated over aachen day training set
		img_mean = [0.3639, 0.3639, 0.3639]
		img_std = [0.2074, 0.2074, 0.2074]

		if training:
			clr_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=[0,0])
		else:
			clr_jitter = transforms.ColorJitter(saturation=[0,0])

		self.image_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(self.imsize),
			clr_jitter,
			transforms.ToTensor(),
			transforms.Normalize(
				mean=img_mean, 
				std=img_std
				)
			])

		self.pose_files = os.listdir(pose_dir)
		self.pose_files = [pose_dir + f for f in self.pose_files]
		self.pose_files.sort()

		self.calibration_files = os.listdir(calibration_dir)
		self.calibration_files = [calibration_dir + f for f in self.calibration_files]
		self.calibration_files.sort()

		if self.training:

			self.init_files = os.listdir(init_dir)
			self.init_files = [init_dir + f for f in self.init_files]
			self.init_files.sort()

			# cluster the dataset
			self.cam_centers, self.cam_sizes, self.labels = self.__cluster__(root_dir, num_clusters)

			print("Calculating ground truth gating probabilities.")
			self.gating_probs = torch.zeros((len(self.init_files), num_clusters))

			for i, f in enumerate(self.init_files):

				init_data = torch.load(f)
				init_data = init_data.view(3, -1)
				init_mask = init_data.sum(0) != 0
				init_data = init_data[:,init_mask]
				init_data = init_data.sum(1) / init_mask.sum()

				init_data = init_data.unsqueeze(0).expand(self.cam_centers.size())
				init_data = init_data - self.cam_centers
				init_data = init_data.norm(dim=1)

				init_data = init_data**2
				init_data = init_data / self.cam_sizes[:,0] / 2
				init_data = torch.exp(-init_data * self.softness)
				init_data /= torch.sqrt(2 * math.pi * self.cam_sizes[:,0])
				init_data /= init_data.sum() + 0.0000001

				self.gating_probs[i] = init_data
	
			if cluster >= 0:
				self.img_sampler = torch.distributions.categorical.Categorical(probs = self.gating_probs[:,self.cluster])
				
			print("Done.")

	def __len__(self):
		'''Returns length of the dataset.'''
		return len(self.rgb_files)

	def get_file_name(self, idx):
		return self.rgb_files[idx]

	def __getitem__(self, idx):
		'''Loads and returns the i th data item of the dataset.

		Returns:
			image: RGB iamge.
			pose: ground truth camera pose.
			init: ground truth scene coordinates.
			seg: binary segmentation that masks clutter: people, sky, cars etc. Obtained via a off-the shelf semantic segmentation method.
			focal_length: Focal length of the camera.
			idx: Pass through of the data item index.
		'''

		if self.cluster >= 0:
			idx = int(self.img_sampler.sample())

		image = io.imread(self.rgb_files[idx])

		if len(image.shape) < 3: # covnert gray scale images
			image = color.gray2rgb(image)

		# scale image and focallength correspondingly
		image_scale = self.imsize / min(image.shape[0:2])
		image = self.image_transform(image)	
	
		focallength = float(numpy.loadtxt(self.calibration_files[idx]))
		focallength *= image_scale

		gt_pose = numpy.loadtxt(self.pose_files[idx])
		gt_pose = torch.from_numpy(gt_pose).float()

		if self.training:
			gt_coords = torch.load(self.init_files[idx])
		else:
			gt_coords = 0

		return idx, image, focallength, gt_pose, gt_coords, -1
