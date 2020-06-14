import torch

import esac

import time
import argparse
import math

from expert import Expert
from expert_ensemble import ExpertEnsemble
import util

import cv2
import numpy as np

parser = argparse.ArgumentParser(
	description='Test ESAC.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', '-m', default='',
	help='ensemble model file, if empty we use the default file name + the session ID')

parser.add_argument('--testinit', '-tinit', action='store_true',
	help='load individual expert networks and gating, used for testing before end-to-end training, we use the default file names + session ID')

parser.add_argument('--testrefined', '-tref', action='store_true',
	help='load individual refined expert networks and gating, used for testing before end-to-end training, we use the default file names + session ID + refined post fix')

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

parser.add_argument('--rotthreshold', '-rt', type=float, default=5, 
	help='acceptance threshold of rotation error in degree')

parser.add_argument('--transthreshold', '-tt', type=float, default=5, 
	help='acceptance threshold of translation error in centimeters')

parser.add_argument('--expertselection', '-es', action='store_true',
	help='select one expert instead of distributing hypotheses')

parser.add_argument('--oracleselection', '-os', action='store_true',
	help='always select the ground truth expert')

parser.add_argument('--clusters', '-c', type=int, default=-1,
	help='number of clusters the environment should be split into, corresponds to the number of desired experts')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()

if opt.clusters < 0:

	# === pre-clustered environment according to environment file ===========

	from room_dataset import RoomDataset
	testset = RoomDataset("test", training=False)
	ensemble = ExpertEnsemble(testset.num_experts)

else:

	# === large, connected environment, perform clustering ==================

	from cluster_dataset import ClusterDataset
	testset = ClusterDataset("test", num_clusters=opt.clusters, training=False)
	ensemble = ExpertEnsemble(testset.num_experts, gating_capacity=2) # for clustering environments we use a gating network with higher capacity

testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6)

if opt.testrefined:

	# load individual, refined experts
	ensemble.load_experts(opt.session, True)

elif opt.testinit:

	# load individual (unrefined) experts
	ensemble.load_experts(opt.session, False)
else:

	# load esac ensemble file
	if len(opt.model) == 0:
		if opt.expertselection:
			opt.model = 'es_%s.net' % opt.session
		else:
			opt.model = 'esac_%s.net' % opt.session

	ensemble.load_ensemble(opt.model)

ensemble.eval()

if opt.testinit:
	opt.session = 'init_' + opt.session

if opt.testrefined:
	opt.session = 'ref_' + opt.session

if opt.expertselection:
	opt.session = 'es_' + opt.session

if opt.oracleselection:
	opt.session = 'os_' + opt.session

test_log = open('results_esac_%s.txt' % (opt.session), 'w', 1)
pose_log = open('poses_esac_%s.txt' % (opt.session), 'w', 1)

print('Environment has', len(testset), 'test images.')

scenes_r = [[]] # hold rotation errors
scenes_t = [[]] # hold translation errors
scenes_c = [[]] # hold classification errors

if opt.clusters < 0:
	for e in range(testset.num_experts-1):
		scenes_r.append([])
		scenes_t.append([])
		scenes_c.append([])

avg_active = 0
max_active = 0
avg_time = 0

with torch.no_grad():	

	for idx, image, focallength, gt_pose, gt_coords, gt_expert in testset_loader:

		idx = int(idx[0])
		img_file = testset.get_file_name(idx)

		print("Processing image %d: %s\n" % (idx, img_file))

		# camera calibration
		focallength = float(focallength[0])
		pp_x = float(image.size(3) / 2)
		pp_y = float(image.size(2) / 2)

		gt_pose = gt_pose[0]
		gt_expert = int(gt_expert[0])

		# dimension of the expert prediction
		pred_w = math.ceil(image.size(3) / Expert.OUTPUT_SUBSAMPLE)
		pred_h = math.ceil(image.size(2) / Expert.OUTPUT_SUBSAMPLE)
		
		# prediction container to hold all expert outputs
		prediction = torch.zeros((testset.num_experts, 3, pred_h, pred_w)).cuda()
		image = image.cuda()

		start_time = time.time()

		# gating prediction
		gating_log_probs = ensemble.log_gating(image)
		gating_probs = torch.exp(gating_log_probs).cpu()

		if opt.oracleselection:
			gating_probs[0].fill_(0)
			gating_probs[0,gt_expert] = 1

		# assign hypotheses to experts
		if opt.expertselection or opt.oracleselection:
			expert = torch.multinomial(gating_probs[0], 1, replacement=True)
			e_hyps = expert.expand((opt.hypotheses))
		else:
			e_hyps = torch.multinomial(gating_probs[0], opt.hypotheses, replacement=True)
		
		# do experts prediction if they have at least one hypothesis
		e_hyps_hist = torch.histc(e_hyps.float(), bins=testset.num_experts, min=0, max=testset.num_experts-1)

		avg_active += float((e_hyps_hist > 0).sum())
		max_active = max(max_active, float((e_hyps_hist > 0).sum()))

		for e, count in enumerate(e_hyps_hist):
			if count > 0:
				prediction[e] = ensemble.scene_coordinates(e, image)

		prediction = prediction.cpu()

		out_pose = torch.zeros(4, 4).float()

		# perform pose estimation
		winning_expert = esac.forward(
			prediction,
			e_hyps, 
			out_pose, 
			0, 
			0, 
			focallength, 
			pp_x,
			pp_y,
			opt.threshold,			
			opt.inlieralpha,
			opt.inlierbeta,
			opt.maxreprojection,
			Expert.OUTPUT_SUBSAMPLE)

		avg_time += time.time()-start_time

		# calculate pose errors
		t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

		gt_R = gt_pose[0:3,0:3].numpy()
		out_R = out_pose[0:3,0:3].numpy()

		r_err = np.matmul(out_R, np.transpose(gt_R))
		r_err = cv2.Rodrigues(r_err)[0]
		r_err = np.linalg.norm(r_err) * 180 / math.pi

		print("\nRotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err*100))

		print("\nTrue expert:", int(gt_expert))
		print("Expert with max. prob.:", int(gating_probs[0].max(0)[1]))
		print("Expert chosen:", winning_expert, "\n")

		scenes_r[gt_expert].append(r_err)
		scenes_t[gt_expert].append(t_err * 100)
		scenes_c[gt_expert].append(int(gt_expert) == winning_expert)


		# write estimated pose to pose file
		out_pose = out_pose.inverse()

		t = out_pose[0:3, 3]

		# rotation to axis angle
		rot, _ = cv2.Rodrigues(out_pose[0:3,0:3].numpy())
		angle = np.linalg.norm(rot)
		axis = rot / angle

		# axis angle to quaternion
		q_w = math.cos(angle * 0.5)
		q_xyz = math.sin(angle * 0.5) * axis

		pose_log.write("%s %f %f %f %f %f %f %f\n" % (
			util.strip_file_name(img_file),
			q_w, q_xyz[0], q_xyz[1], q_xyz[2],
			float(t[0]), float(t[1]), float(t[2])))	

# calculate overall evaluation statistics
print("Scene - Class.Acc. - Pose.Acc. - Median Rot. - Median Trans.")
print("------------------------------------------------------------")

avg_class = 0
avg_pose = 0
avg_rot = 0
avg_trans = 0

for sceneIdx in range(len(scenes_c)):

	class_acc = sum(scenes_c[sceneIdx])/max(len(scenes_c[sceneIdx]),1)
	avg_class += class_acc

	pose_acc = [(t_err < opt.transthreshold and r_err < opt.rotthreshold) for (t_err, r_err) in zip(scenes_t[sceneIdx], scenes_r[sceneIdx])]
	pose_acc = sum(pose_acc)/max(len(pose_acc),1)
	avg_pose += pose_acc

	def median(l):
		if len(l) == 0: 
			return 0
		l.sort()
		return l[int(len(l) / 2)]

	median_r = median(scenes_r[sceneIdx])
	avg_rot += median_r

	median_t = median(scenes_t[sceneIdx])
	avg_trans += median_t

	print("%7d %7.1f%% %10.1f%% %10.2fdeg %10.2fcm" % (sceneIdx, class_acc*100, pose_acc*100, median_r, median_t))
	test_log.write("%f %f %f %f\n" % (class_acc, pose_acc, median_r, median_t))

if opt.clusters < 0:
	print("------------------------------------------------------------")
	print("Average %7.1f%% %10.1f%% %10.2fdeg %10.2fcm" % (avg_class*100/testset.num_experts, avg_pose*100/testset.num_experts, avg_rot/testset.num_experts, avg_trans/testset.num_experts))

print('\nAvg. experts active:', avg_active / len(testset_loader))
print('Max. experts active:', max_active)

print("\nAvg. Time: %.3fs" % (avg_time / len(testset_loader)))

print('\nDone without errors.')
test_log.close()
pose_log.close()


