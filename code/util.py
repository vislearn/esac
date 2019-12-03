import torch.nn as nn
import random

def random_shift(image, max_shift):
	''' Randomly shift the input image via zero padding in x and y. '''

	padX = random.randint(-max_shift, max_shift)
	padY = random.randint(-max_shift, max_shift)
	pad = nn.ZeroPad2d((padX, -padX, padY, -padY))

	return padX, padY, pad(image)

def clamp_tensor(coords1, coords2):
	''' Clamp coords1 to height and width of coords2. '''

	return coords1[:,:,0:coords2.size(2),0:coords2.size(3)]

def assert_size(coords1, coords2):
	''' Asserts that both tensors have the same width and height, except for a toleranec of 1px. '''
	
	delta_h = coords1.size(2)-coords2.size(2)
	delta_w = coords1.size(3)-coords2.size(3)

	if abs(delta_h) > 1 or abs(delta_w) > 1:
		print("Error: Tensor size mismatch!")
		print(coords1.size())
		print(coords2.size())
		exit()

	if delta_h > 0 or delta_w > 0:
		coords1 = clamp_tensor(coords1, coords2)

	if delta_h < 0 or delta_w < 0:
		coords2 = clamp_tensor(coords2, coords1)

	return coords1, coords2

def clamp_probs(probs, n):
	''' Sets all entries in probs to zero except the n biggest ones. '''

	if n < 0: return

	s_prob, s_indx = probs.sort(dim=0)

	for i, idx in enumerate(s_indx):
		if i < s_prob.size(0) - n:
			probs[idx] = 0	

def strip_file_name(f):
	''' Removes path and prefixes related to the Aachen dataset. '''

	# remove path
	f = f.split('/')[-1]

	# remove aachen prefixes
	aachen_ignore_list = ['db_', 'query_day_milestone_', 'query_day_nexus4_', 'query_day_nexus5x_', 'query_night_nexus5x_']
	for ign in aachen_ignore_list:
		if f.startswith(ign):
			f = f[len(ign):]
	return f
