import os
import numpy
import math

from skimage import io
from skimage import color

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from random import randint
import random

class RoomDataset(Dataset):
	"""Camera re-localization dataset for environments consisting of separate rooms.

	The list of rooms will be read from a env_list file.
	This file should contain information for each room per line, 
	namely the path to the dataset directory, 
	and (optionally) the center of the room in world coordinates (x, y, z).
	
	"""

	def __init__(self, root_dir, scene = -1, training=True, imsize=480, normalize_mean=True, grid_cell_size=5):

		self.training = training
		self.imsize = imsize
		self.grid_cell_size = grid_cell_size
		self.normalize_mean = normalize_mean

		with open('env_list.txt', 'r') as f:
			environment = f.readlines()

		self.scenes = []
		self.means = torch.zeros((len(environment), 3))

		for i, line in enumerate(environment):
			line = line.split()
			self.scenes.append(line[0])

			# check wehther center coordinate have been provided for normalization
			if len(line) > 1:
				self.means[i, 0] = float(line[1])
				self.means[i, 1] = float(line[2])
				self.means[i, 2] = float(line[3])

		self.num_experts = len(self.scenes)

		rgb_dir = root_dir + '/rgb/'
		pose_dir =  root_dir + '/poses/'
		calibration_dir = root_dir + '/calibration/'
		init_dir =  root_dir + '/init/'
		
		self.rgb_files = {}
		self.pose_files = {}
		self.calibration_files = {}
		self.init_files = {}

		self.scene = scene

		self.scenes_cnt = []
		self.cnt = 0

		# read files
		for scene in self.scenes:

			cur_rgb = os.listdir(scene + '/' + rgb_dir)
			cur_rgb = [scene + '/' + rgb_dir + f for f in cur_rgb]
			cur_rgb.sort()
			self.rgb_files[scene] = cur_rgb

			self.scenes_cnt.append(self.cnt)
			self.cnt += len(cur_rgb)

			cur_pose = os.listdir(scene + '/' + pose_dir)
			cur_pose = [scene + '/' + pose_dir + f for f in cur_pose]
			cur_pose.sort()
			self.pose_files[scene] = cur_pose

			cur_calibration = os.listdir(scene + '/' + calibration_dir)
			cur_calibration = [scene + '/' + calibration_dir + f for f in cur_calibration]
			cur_calibration.sort()	
			self.calibration_files[scene] = cur_calibration

			if self.training:
				cur_init = os.listdir(scene + '/' + init_dir)
				cur_init = [scene + '/' + init_dir + f for f in cur_init]
				cur_init.sort()
				self.init_files[scene] = cur_init

		self.image_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(self.imsize),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.4, 0.4, 0.4], # statistics calculated over 7scenes training set
				std=[0.25, 0.25, 0.25]
				)
			])

		self.pose_transform = transforms.Compose([
			transforms.ToTensor()
			])

	def __len__(self):

		if self.scene >= 0:
			return len(self.rgb_files[self.scenes[self.scene]])
		else:

			if self.training:
				return 1000
			else:
				return self.cnt

	def global_to_local_idx(self, idx):
		''' map global idx to scene and scene idx '''

		for i, cnt in enumerate(self.scenes_cnt):
			if idx < cnt:
				scene = self.scenes[i-1]
				idx -= self.scenes_cnt[i-1]
				break
			elif i == len(self.scenes_cnt) - 1:
				scene = self.scenes[i]
				idx -= self.scenes_cnt[i]		

		return scene, idx

	def get_file_name(self, global_idx):
		''' return rgb file name corresponding to index '''

		scene, local_idx = self.global_to_local_idx(global_idx)
		return self.rgb_files[scene][local_idx]

	def __getitem__(self, global_idx):

		# index mapping
		if self.scene >= 0:
			# index refers to current scene
			local_idx = global_idx
			scene = self.scenes[self.scene]
		else:

			# global environment index
			if self.training:
				# random image of random scene
				scene = random.choice(self.scenes)
				local_idx = randint(0, len(self.rgb_files[scene]) - 1)
			else:
				scene,local_idx = self.global_to_local_idx(global_idx)

		# load image
		image = io.imread(self.rgb_files[scene][local_idx])

		if len(image.shape) < 3:
			image = color.gray2rgb(image)
		
		# scale image and focallength correspondingly
		image_scale = self.imsize / min(image.shape[0:2])
		image = self.image_transform(image)	

		focallength = float(numpy.loadtxt(self.calibration_files[scene][local_idx]))
		focallength *= image_scale

		# load pose
		gt_pose = numpy.loadtxt(self.pose_files[scene][local_idx])
		gt_pose = torch.from_numpy(gt_pose).float()

		# map local scene coordinate system to global environment coordinate system
		scene_idx = self.scenes.index(scene)

		# normalize scene coordinate system (zero mean)
		offset = self.means[scene_idx].clone()
		if not self.normalize_mean: offset.fill_(0)

		# shift scene within grid
		grid_size = math.ceil(math.sqrt(len(self.scenes)))

		row = math.ceil((scene_idx+1) / grid_size)-1
		col = scene_idx % grid_size

		offset[0] += row * self.grid_cell_size
		offset[1] += col * self.grid_cell_size

		# shift ground truth pose 	
		gt_pose[0:3, 3] -= offset.float()

		# shift ground truth scene coordinates	
		if self.training:

			gt_coords = torch.load(self.init_files[scene][local_idx])
			gt_coords_size = gt_coords.size()
			gt_coords = gt_coords.view(3, -1)

			# do not shift invalid coordinates (all zeros)			
			coords_mask = gt_coords.abs().sum(0) == 0 

			# shift
			offset = offset.unsqueeze(1).expand(gt_coords.size())
			gt_coords = gt_coords - offset.float()

			# reset invalid coordinates to zero
			if coords_mask.sum() > 0:
				gt_coords[:, coords_mask] = 0

			gt_coords = gt_coords.view(gt_coords_size)

		else:

			# no ground truth scene coordinate necessary in test mode
			gt_coords = 0

		return global_idx, image, focallength, gt_pose, gt_coords, self.scenes.index(scene)
