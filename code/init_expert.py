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

parser.add_argument('--cutloss', '-cl', type=float, default=100, help='robust square root loss after this threshold')

opt = parser.parse_args()

if opt.clusters < 0:

	# === pre-clustered environment according to environment file ===========
	from room_dataset import RoomDataset
	trainset = RoomDataset("training", scene=opt.expert)
	trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

	print("Calculating mean scene coordinate...")

	mean = torch.zeros((3,))
	count = 0

	for idx, image, focallength, gt_pose, gt_coords, gt_expert in trainset_loader:

		gt_coords = gt_coords[0]
		gt_coords = gt_coords.view(3, -1)

		coord_mask = gt_coords.abs().sum(0) > 0
		gt_coords = gt_coords[:, coord_mask]

		mean += gt_coords.sum(1)
		count += int(coord_mask.sum())

	mean /= count

	print("Done. Mean: %.2f, %.2f, %.2f\n" % (mean[0], mean[1], mean[2]))

	model = Expert(mean)

else:

	# === large, connected environment, perform clustering ==================
	from cluster_dataset import ClusterDataset
	trainset = ClusterDataset("training", num_clusters=opt.clusters, cluster=opt.expert)
	trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

	model = Expert(trainset.cam_centers[opt.expert])


model.cuda()
model.train()

model_file = 'expert_e%d_%s.net' % (opt.expert, opt.session)

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrssteps, gamma=opt.lrsgamma)

iteration = 0
epochs = math.ceil(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_init_e%d_%s.txt' % (opt.expert, opt.session), 'w', 1)

for epoch in range(epochs):	

	print("=== Epoch: %d ======================================" % epoch)

	torch.save(model.state_dict(), model_file)
	print('Saved model.')

	for idx, image, focallength, gt_pose, gt_coords, gt_expert in trainset_loader:

		start_time = time.time()

		gt_coords = gt_coords.cuda()
		image = image.cuda()

		#random shift as data augmentation
		padX, padY, image = util.random_shift(image, Expert.OUTPUT_SUBSAMPLE / 2)
	
		prediction = model(image)

		prediction, gt_coords = util.assert_size(prediction, gt_coords)
		prediction = prediction.squeeze().contiguous().view(3, -1)
		gt_coords = gt_coords.squeeze().contiguous().view(3, -1)

		# mask out invalid coordinates (all zeros)
		coords_mask = gt_coords.abs().sum(0) != 0 
		prediction = prediction[:,coords_mask]
		gt_coords = gt_coords[:,coords_mask]

		loss = torch.norm(prediction - gt_coords, dim=0)

		# soft clamping of loss for stability
		loss_l1 = loss[loss <= opt.cutloss]
		loss_sqrt = loss[loss > opt.cutloss]
		loss_sqrt = torch.sqrt(opt.cutloss * loss_sqrt)

		robust_loss = (loss_l1.sum() + loss_sqrt.sum()) / float(loss.size(0))

		robust_loss.backward()	# calculate gradients (pytorch autograd)
		optimizer.step()		# update all model parameters
		scheduler.step()
		optimizer.zero_grad()
		
		print('Iteration: %6d, Loss: %.1f, Time: %.2fs' % (iteration, robust_loss, time.time()-start_time), flush=True)
		train_log.write('%d %f\n' % (iteration, robust_loss))

		iteration = iteration + 1

print('Done without errors.')
train_log.close()
