import torch
import torch.optim as optim

from expert import Expert
from gating import Gating

class ExpertEnsemble:

	def __init__(self, num_experts, lr=0, cam_centers=None, gating_capacity=1):
	
		self.num_experts = num_experts
		self.lr = lr # learning rate

		if cam_centers is None:
			cam_centers = torch.zeros(num_experts, 3)

		cam_centers = cam_centers.cuda()

		# setup gating network
		self.model_g = Gating(num_experts, gating_capacity)
		self.model_g = self.model_g.cuda()
		self.model_g.train()
		self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=lr)

		# setup expert networks
		self.experts = []
		self.expert_opts = []
		
		for i in range(0, num_experts):

			model_e = Expert(cam_centers[i])
			model_e = model_e.cuda()
			model_e.train()
			optimizer_e = optim.Adam(model_e.parameters(), lr=lr)

			self.experts.append(model_e)
			self.expert_opts.append(optimizer_e)


	def __call__(self, image, expert):
		''' Simple forward pass. Do gating prediction, and expert prediction for the given expert. '''

		selection = self.model_g(image)
		prediction = self.experts[expert](image)

		return prediction, selection

	def log_gating(self, image):
		''' Return log probabilities for gating. '''

		return self.model_g(image)

	def scene_coordinates(self, expert, image):
		''' Return scene coordinate prediction for a given expert. '''

		return self.experts[expert](image)

	def update(self, e_hyps_hist):
		''' Update parameters of gating and expert networks. '''

		# update parameters
		self.optimizer_g.step()
		self.optimizer_g.zero_grad()

		for e, e_opt in enumerate(self.expert_opts):
			if e_hyps_hist[e] > 0:
				e_opt.step() 
				e_opt.zero_grad() 

	def train(self):
		''' Set all ensemble networks to training mode. '''

		self.model_g.train()
		for expert in self.experts:
			expert.train()

	def eval(self):
		''' Set all ensemble networks to test mode. '''

		self.model_g.eval()
		for expert in self.experts:
			expert.eval()

	def save(self, path):
		''' Save the ensemble. '''

		state_dict = []
		state_dict.append(self.model_g.state_dict())
		for expert in self.experts:
			state_dict.append(expert.state_dict())

		torch.save(state_dict, path)

	def load_experts(self, session, ref=False):
		''' Load the ensemble from individual network files. '''
	
		self.model_g.load_state_dict(torch.load('./gating_%s.net' % session))

		if ref:
			session += '_refined'

		for i, expert in enumerate(self.experts):
			expert.load_state_dict(torch.load('./expert_e%d_%s.net' % (i, session)))
	
	def load_ensemble(self, path):
		''' Load the ensemble from a combined ensemble file. '''

		state_dict = torch.load(path)

		self.model_g.load_state_dict(state_dict[0])
		for i, expert in enumerate(self.experts):
			expert.load_state_dict(state_dict[i+1])

		
