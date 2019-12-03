import os
import argparse
import math

import torch
import cv2

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

parser = argparse.ArgumentParser(
	description='Setup the Dubrovnik dataset.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dbfolder', '-db', type=str, required=True,
	help='relative path to the folder containing the dubrovnik database images')

opt = parser.parse_args()

# check image folder provided by the user
existing_images = os.listdir('./' + opt.dbfolder)
if len(existing_images) != 6844:
	print("ERROR: Expected 6844 images in", opt.dbfolder, "but got", len(existing_images))
	exit()

# name of the folder where we download the original dubrovnik dataset to
src_folder = 'dubrovnik_source'

# destination folder that will contain the dataset in our format
dst_folder = 'dubrovnik'

mkdir(dst_folder)
mkdir(src_folder)
os.chdir(src_folder)

# download the publik portion dubrovnik dataset
print("=== Downloading Dubrovnik Data ===============================")

os.system('wget http://s3.amazonaws.com/LocationRecognition/Datasets/Dubrovnik6K.tar.gz')
os.system('tar -xvzf Dubrovnik6K.tar.gz')
os.system('rm Dubrovnik6K.tar.gz')

#ignore outlier training images
ignore_images = [
	'klazien_2867455427.jpg', 
	'eas8302_2799908943.jpg', 
	'28512365@N08_2967791490.jpg', 
	'neozeitgeist_2614406528.jpg', 
	'22417886@N02_2162265990.jpg', 
	'westius_2275531184.jpg']

def create_set(input_file, input_list, image_list, training):

	if training: 
		variant = 'training'
	else:
		variant = 'test'

	mkdir('../' + dst_folder + '/' + variant + '/rgb')
	mkdir('../' + dst_folder + '/' + variant + '/calibration')
	mkdir('../' + dst_folder + '/' + variant + '/poses')
	if training: mkdir('../' + dst_folder + '/' + variant + '/init')

	f = open(input_file)
	reconstruction = f.readlines()
	f.close()

	f = open(image_list)
	image_list = f.readlines()
	f.close()

	f = open(input_list)
	input_list = f.readlines()
	input_list = [line.strip() for line in input_list]
	f.close()	

	line = reconstruction[1].split()

	num_cams = int(line[0])
	num_pts = int(line[1])

	if training:
		# read 3D points
		pts_dict = {}
		for cam_idx in range(0, num_cams):
			pts_dict[cam_idx] = []


		pt = pts_start = 2 + num_cams * 5
		pts_end = pts_start + num_pts * 3

		while pt < pts_end:

			pt_list = reconstruction[pt+2].split()
			num_views = int(pt_list[0])
			
			#if num_views > 10: # filter instable points
			for pt_view in range(0, num_views):
				cam_view = int(pt_list[1 + pt_view * 4])

				pt_3D = [float(x) for x in reconstruction[pt].split()]
				pt_3D.append(1.0)
				pts_dict[cam_view].append(pt_3D)

			pt += 3


	for cam_idx in range(len(image_list)):

		print("Processing camera %d of %d." % (cam_idx+1, len(image_list)))

		image_file = image_list[cam_idx].split()[0]
		cam_idx = input_list.index(image_file)

		if training:
			image_file = image_file[3:]
		else:
			image_file = image_file[6:]

		if image_file not in existing_images:
			print("Skipping camera. " + image_file + " does not exist.")
			continue
		if image_file in ignore_images:
			print("Skipping camera. " + image_file + " is in ignore list.")
			continue

		#link image
		os.system('ln -s ../../../' + opt.dbfolder + '/' + image_file + ' ../' + dst_folder + '/' + variant + '/rgb/')

		# read cameras
		calibration = [float(c) for c in reconstruction[2+cam_idx*5].split()]
		focallength = calibration[0]

		cam_pose = []
		cam_pose.append([float(x) for x in reconstruction[3+cam_idx*5].split()])
		cam_pose.append([-float(x) for x in reconstruction[4+cam_idx*5].split()])
		cam_pose.append([-float(x) for x in reconstruction[5+cam_idx*5].split()])
		cam_pose.append([0.0, 0.0, 0.0, 1.0])

		cam_trans = [float(x) for x in reconstruction[6+cam_idx*5].split()]
		cam_pose[0].append(cam_trans[0])
		cam_pose[1].append(-cam_trans[1])
		cam_pose[2].append(-cam_trans[2])

		cam_pose = torch.tensor(cam_pose)

		if training:
			pts_3D = torch.tensor(pts_dict[cam_idx])

			#load image for image dimensions
			image = cv2.imread('../' + opt.dbfolder + '/' + image_file)

			img_aspect = image.shape[0] / image.shape[1]
			target_height = 60

			if img_aspect > 1:
				#portrait
				out_w = target_height
				out_h = int(math.ceil(target_height * img_aspect))	
			else:
				#landscape
				out_w = int(math.ceil(target_height / img_aspect))
				out_h = target_height

			out_scale = out_w / image.shape[1]
			out_tensor = torch.zeros((3, out_h, out_w))
			out_zbuffer = torch.zeros((out_h, out_w))

			# render 3D points with basic depth test
			for pt_idx in range(pts_3D.size(0)):

				scene_pt = pts_3D[pt_idx]
				scene_pt = scene_pt.unsqueeze(0)
				scene_pt = scene_pt.transpose(0, 1)

				cam_pt = torch.mm(cam_pose, scene_pt)

				img_pt = cam_pt[0:2, 0] * focallength / cam_pt[2, 0] * out_scale

				y = img_pt[1] + out_h / 2
				y = int(torch.clamp(y, min=0, max=out_h-1))

				x = img_pt[0] + out_w / 2
				x = int(torch.clamp(x, min=0, max=out_w-1))

				if out_zbuffer[y, x] == 0 or out_zbuffer[y, x] > cam_pt[2, 0]:
					out_zbuffer[y, x] = cam_pt[2, 0]
					out_tensor[:, y, x] = pts_3D[pt_idx, 0:3]

			if out_zbuffer.sum() == 0:
				print("Skip camera: Empty zbuffer!")
				continue

			torch.save(out_tensor, '../' + dst_folder + '/' + variant + '/init/' + image_file[:-4] + '.dat')

		with open('../' + dst_folder + '/' + variant + '/calibration/' + image_file[:-3] + 'txt', 'w') as f:
			f.write(str(focallength))

		cam_pose = cam_pose.inverse()

		with open('../' + dst_folder + '/' + variant + '/poses/' + image_file[:-3] + 'txt', 'w') as f:
			f.write(str(float(cam_pose[0, 0])) + ' ' + str(float(cam_pose[0, 1])) + ' ' + str(float(cam_pose[0, 2])) + ' ' + str(float(cam_pose[0, 3])) + '\n')
			f.write(str(float(cam_pose[1, 0])) + ' ' + str(float(cam_pose[1, 1])) + ' ' + str(float(cam_pose[1, 2])) + ' ' + str(float(cam_pose[1, 3])) + '\n')
			f.write(str(float(cam_pose[2, 0])) + ' ' + str(float(cam_pose[2, 1])) + ' ' + str(float(cam_pose[2, 2])) + ' ' + str(float(cam_pose[2, 3])) + '\n')
			f.write(str(float(cam_pose[3, 0])) + ' ' + str(float(cam_pose[3, 1])) + ' ' + str(float(cam_pose[3, 2])) + ' ' + str(float(cam_pose[3, 3])) + '\n')

create_set('Dubrovnik6K/bundle/bundle.db.out', 'Dubrovnik6K/list.db.txt', 'Dubrovnik6K/list.db.txt', training=True)
create_set('Dubrovnik6K/bundle/bundle.orig.out', 'Dubrovnik6K/bundle/list.orig.txt', 'Dubrovnik6K/list.query.txt', training=False)