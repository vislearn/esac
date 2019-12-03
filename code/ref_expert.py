import torch
import torch.nn as nn
import torch.optim as optim

import time
import argparse
import math
import random

from expert import Expert
import util

import numpy as np

parser = argparse.ArgumentParser(
	description='Train large scale camera localization.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--expert', '-e', type=int, default=0, 
	help='expert to train')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--iterations', '-iter', type=int, default=1000000,
	help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--lrssteps', '-lrss', type=int, default=400000, help='step size for learning rate schedule')

parser.add_argument('--lrsgamma', '-lrsg', type=float, default=0.5, help='discount factor of learning rate schedule')

parser.add_argument('--clusters', '-c', type=int, default=-1,
	help='number of clusters the environment should be split into, corresponds to the number of desired experts')

parser.add_argument('--cutloss', '-cl', type=float, default=10, help='robust square root loss after this threshold')


opt = parser.parse_args()

if opt.clusters < 0:

	# === pre-clustered environment according to environment file ===========
	from dataset import RoomDataset
	trainset = RoomDataset("training", scene=opt.expert)

else:

	# === large, connected environment, perform clustering ==================
	from cluster_dataset import ClusterDataset
	trainset = ClusterDataset("training", num_clusters=opt.clusters, cluster=opt.expert)

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

model = Expert(torch.zeros((3,)))
model.load_state_dict(torch.load('expert_e%d_%s.net' % (opt.expert, opt.session)))

print("Successfully loaded model.")

model.cuda()
model.train()

model_file = 'expert_e%d_%s_refined.net' % (opt.expert, opt.session)

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrssteps, gamma=opt.lrsgamma)

iteration = 0
epochs = math.ceil(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_refine_e%d_%s.txt' % (opt.expert, opt.session), 'w', 1)

# generate grid of target reprojection pixel positions
prediction_grid = torch.zeros((2, 
	math.ceil(5000 / Expert.OUTPUT_SUBSAMPLE),  # 5000px is max limit of image size, increase if needed
	math.ceil(5000 / Expert.OUTPUT_SUBSAMPLE)))

for x in range(0, prediction_grid.size(2)):
	for y in range(0, prediction_grid.size(1)):
		prediction_grid[0, y, x] = x * Expert.OUTPUT_SUBSAMPLE + Expert.OUTPUT_SUBSAMPLE / 2
		prediction_grid[1, y, x] = y * Expert.OUTPUT_SUBSAMPLE + Expert.OUTPUT_SUBSAMPLE / 2

prediction_grid = prediction_grid.cuda()

for epoch in range(epochs):	

	print("=== Epoch: %d ======================================" % epoch)

	torch.save(model.state_dict(), model_file)
	print('Saved model.')

	for idx, image, focallength, gt_pose, gt_coords, gt_expert in trainset_loader:

		start_time = time.time()

		image = image.cuda()
		padX, padY, image = util.random_shift(image, Expert.OUTPUT_SUBSAMPLE / 2)
	
		prediction = model(image)

		# apply random shift to the ground truth reprojection positions as well
		prediction_grid_pad = prediction_grid[:,0:prediction.size(2),0:prediction.size(3)].clone()
		prediction_grid_pad = prediction_grid_pad.view(2, -1)

		prediction_grid_pad[0] -= padX
		prediction_grid_pad[1] -= padY

		# create camera calibartion matrix
		focallength = float(focallength[0])
		cam_mat = torch.eye(3)
		cam_mat[0, 0] = focallength
		cam_mat[1, 1] = focallength
		cam_mat[0, 2] = image.size(3) / 2
		cam_mat[1, 2] = image.size(2) / 2
		cam_mat = cam_mat.cuda()

		# predicted scene coordinates to homogeneous coordinates
		ones = torch.ones((prediction.size(0), 1, prediction.size(2), prediction.size(3)))
		ones = ones.cuda()
		prediction = torch.cat((prediction, ones), 1)

		gt_pose = gt_pose[0].inverse()[0:3,:]
		gt_pose = gt_pose.cuda()

		# scene coordinate to camera coordinate 
		prediction = prediction[0].view(4, -1)
		eye = torch.mm(gt_pose, prediction)

		# image reprojection
		px = torch.mm(cam_mat, eye)
		px[2].clamp_(min=0.1) #avoid division by zero
		px = px[0:2] / px[2]

		# reprojection error
		px = px - prediction_grid_pad
		px = px.norm(2, 0)
		px = px.clamp(0, 100) # reprojection error beyond 100px is not useful

		loss_l1 = px[px <= opt.cutloss]
		loss_sqrt = px[px > opt.cutloss]
		loss_sqrt = torch.sqrt(opt.cutloss * loss_sqrt)

		robust_loss = (loss_l1.sum() + loss_sqrt.sum()) / float(px.size(0))

		robust_loss.backward()	# calculate gradients (pytorch autograd)
		optimizer.step()		# update all model parameters
		scheduler.step()
		optimizer.zero_grad()
		
		print('Iteration: %6d, Loss: %.1f, Time: %.2fs' % (iteration, robust_loss, time.time()-start_time), flush=True)
		train_log.write('%d %f\n' % (iteration, robust_loss))

		iteration = iteration + 1

print('Done without errors.')
train_log.close()