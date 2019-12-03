import torch
import torch.nn as nn
import torch.nn.functional as F



class Expert(nn.Module):
	'''
	FCN architecture for large scale scene coordiante regression.

	'''

	OUTPUT_SUBSAMPLE = 8

	def __init__(self, mean):
		'''
		Constructor.
		'''
		super(Expert, self).__init__()

		self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
		self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

		self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
		self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
		self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

		self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
		self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res2_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

		self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

		self.res3_conv1 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res3_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
		self.res3_conv3 = nn.Conv2d(512, 512, 1, 1, 0)

		self.fc1 = nn.Conv2d(512, 512, 1, 1, 0)
		self.fc2 = nn.Conv2d(512, 512, 1, 1, 0)
		self.fc3 = nn.Conv2d(512, 3, 1, 1, 0)

		# register the center coordinate for de-normalization of the output
		self.register_buffer('mean', torch.tensor(mean.size()).cuda())
		self.mean = mean.clone()

	def forward(self, inputs):
		'''
		Forward pass.

		inputs -- 4D data tensor (BxCxHxW)
		'''

		x = inputs
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		res = F.relu(self.conv4(x))

		x = F.relu(self.res1_conv1(res))
		x = F.relu(self.res1_conv2(x))
		x = F.relu(self.res1_conv3(x))
		
		res = res + x

		x = F.relu(self.res2_conv1(res))
		x = F.relu(self.res2_conv2(x))
		x = F.relu(self.res2_conv3(x))

		res = self.res2_skip(res) + x

		x = F.relu(self.res3_conv1(res))
		x = F.relu(self.res3_conv2(x))
		x = F.relu(self.res3_conv3(x))

		res = res + x		

		x = F.relu(self.fc1(res))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		
		x[:, 0] += self.mean[0]
		x[:, 1] += self.mean[1]
		x[:, 2] += self.mean[2]

		return x
