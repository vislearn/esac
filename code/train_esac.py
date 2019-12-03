import torch

import time
import argparse
import math

import esac

from expert import Expert
from expert_ensemble import ExpertEnsemble
import util

import cv2
import numpy as np

parser = argparse.ArgumentParser(
	description='Train an ensemble of experts end-to-end, minimizing the final pose error.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--hypotheses', '-hyps', type=int, default=256, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; Controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--inlierbeta', '-ib', type=float, default=0.5, 
	help='beta parameter of the soft inlier count; controls the softness of the sigmoid; lower means softer')

parser.add_argument('--maxreprojection', '-maxr', type=float, default=100, 
	help='maximum reprojection error; reprojection error is clamped to this value for stability')

parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
	help='learning rate')

parser.add_argument('--iterations', '-iter', type=int, default=50000,
	help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
	help='weight of rotation error (in degree')

parser.add_argument('--weighttrans', '-wt', type=float, default=100.0, 
	help='weight of translation error (in meters)')

parser.add_argument('--losscut', '-lc', type=float, default=100.0, 
	help='soft clamping (sqare root of loss) after this value')

parser.add_argument('--clusters', '-c', type=int, default=-1,
	help='number of clusters the environment should be split into, corresponds to the number of desired experts')

parser.add_argument('--maxexperts', '-maxe', type=int, default=-1, 
	help='restricts the maximum number of experts evaluated per image, -1 means no restriction')

parser.add_argument('--expertselection', '-es', action='store_true',
	help='select one expert instead of distributing hypotheses')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--refined', '-ref', action='store_true',
	help='load refined experts')

opt = parser.parse_args()

if opt.clusters < 0:

	# === pre-clustered environment according to environment file ===========

	from room_dataset import RoomDataset
	trainset = RoomDataset("training", training=True)
	ensemble = ExpertEnsemble(trainset.num_experts, lr=opt.learningrate)

else:

	# === large, connected environment, perform clustering ==================

	from cluster_dataset import ClusterDataset
	trainset = ClusterDataset("training", num_clusters=opt.clusters, training=True)
	ensemble = ExpertEnsemble(trainset.num_experts, lr=opt.learningrate, gating_capacity=2) # for clustering environments we use a gating network with higher capacity


trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)
ensemble.load_experts(opt.session, opt.refined)

if opt.expertselection:
	model_file = 'es_%s.net' % (opt.session)
	train_log = open('log_es_%s.txt' % (opt.session), 'w', 1)
else:
	model_file = 'esac_%s.net' % (opt.session)
	train_log = open('log_esac_%s.txt' % (opt.session), 'w', 1)


iteration = 0
epochs = math.ceil(opt.iterations / len(trainset))

for epoch in range(epochs):	

	print("=== Epoch: %d ======================================" % epoch)

	ensemble.save(model_file)
	print('Saved model.')
	
	for idx, image, focallength, gt_pose, gt_coords, gt_expert in trainset_loader:

		start_time = time.time()

		# camera calibration
		pp_x = float(image.size(3) / 2)
		pp_y = float(image.size(2) / 2)
		focallength = float(focallength[0])

		gt_pose = gt_pose[0]	

		# dimension of the expert prediction
		pred_w = math.ceil(image.size(3) / Expert.OUTPUT_SUBSAMPLE)
		pred_h = math.ceil(image.size(2) / Expert.OUTPUT_SUBSAMPLE)

		# prediction container to hold all expert outputs
		prediction = torch.zeros((trainset.num_experts, 3, pred_h, pred_w)).cuda()
		image = image.cuda()

		#random shift as data augmentation
		padX, padY, image = util.random_shift(image, Expert.OUTPUT_SUBSAMPLE / 2)

		# gating prediction
		gating_log_probs = ensemble.log_gating(image)
		gating_probs = torch.exp(gating_log_probs).cpu()

		util.clamp_probs(gating_probs[0], opt.maxexperts)

		# assign hypotheses to experts
		if opt.expertselection:
			expert = torch.multinomial(gating_probs[0], 1, replacement=True)
			e_hyps = expert.expand((opt.hypotheses))
		else:
			e_hyps = torch.multinomial(gating_probs[0], opt.hypotheses, replacement=True)

		# do experts prediction if they have at least one hypothesis
		e_hyps_hist = torch.histc(e_hyps.float(), bins=trainset.num_experts, min=0, max=trainset.num_experts-1)

		for e, count in enumerate(e_hyps_hist):
			if count > 0:
				prediction[e] = ensemble.scene_coordinates(e, image)

		# gradient container to be filled by the C++ extension
		prediction_gradients = torch.zeros(prediction.size())

		# perform pose estimation and calculate expert gradients
		loss = esac.backward(
			prediction.cpu(),
			prediction_gradients,
			e_hyps, 
			gt_pose, 
			opt.weightrot,
			opt.weighttrans,
			opt.losscut,			
			padX, 
			padY, 
			focallength, 
			pp_x,
			pp_y,
			opt.threshold,			
			opt.inlieralpha,
			opt.inlieralpha,
			opt.maxreprojection,
			Expert.OUTPUT_SUBSAMPLE)

		# calculate gating gradients
		if opt.expertselection:
			gating_log_prob_gradients = torch.zeros(gating_log_probs.size())
			gating_log_prob_gradients[0, int(expert)] = loss
		else:
			gating_log_prob_gradients = loss * e_hyps_hist
			gating_log_prob_gradients.unsqueeze_(0)

		torch.autograd.backward(
			(prediction, gating_log_probs), 
			(prediction_gradients.cuda(), gating_log_prob_gradients.cuda()))

		# update all model parameters		
		ensemble.update(e_hyps_hist) 
		
		print('Iteration: %6d, Loss: %.2f, Time: %.2fs \n' % (iteration, loss, time.time()-start_time), flush=True)
		train_log.write('%d %f \n' % (iteration, loss))

		iteration = iteration + 1

print('Done without errors.')
train_log.close()
