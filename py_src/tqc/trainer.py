import torch
import os
from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE
import rotations
import tools

class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		top_quantiles_to_drop,
		target_entropy,
		lr=3e-4
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)
		self.lr = lr
		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy

		self.quantiles_total = critic.n_quantiles * critic.n_nets

		self.total_it = 0

	def train(self, replay_buffer, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)

		# --- Critic Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)

		# action_loss = torch.mean(torch.abs(new_next_action-new_action))
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean() #+ (new_next_action-new_next_action).mean()
		# actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean() + action_loss
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

		# --- Actor Update ---
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




		self.total_it += 1


	def train_with_demo(self, replay_buffer, replay_buffer_expert, batch_size=256):
		sample_policy = replay_buffer.sample(int(batch_size/2))
		sample_demo = replay_buffer_expert.sample(int(batch_size/2))
		state, action, next_state, reward, not_done = self.concatenate_samples(sample_policy, sample_demo)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)

		# --- Critic Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)

		# action_loss = torch.mean(torch.abs(new_next_action-new_action))
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean() #+ (new_next_action-new_next_action).mean()
		# actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean() + action_loss
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

		# --- Actor Update ---
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




		self.total_it += 1

	def save(self, filename):

		filename = str(filename)
		os.makedirs(filename, exist_ok=True)
		if not os.path.exists(filename):
			os.makedirs(filename)

		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.log_alpha, filename + '_log_alpha')
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_alpha = torch.load(filename + '_log_alpha')
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))

	def concatenate_samples(self, sample1, sample2):
		return tuple(torch.cat((s1, s2), dim=0) for s1, s2 in zip(sample1, sample2))


class Custom_Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		top_quantiles_to_drop,
		target_entropy,
		lr=3e-4
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)
		self.lr = lr
		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy

		self.quantiles_total = critic.n_quantiles * critic.n_nets

		self.total_it = 0

	def train(self, replay_buffer, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)

		# --- Critic Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)

		# action_loss = torch.mean(torch.abs(new_next_action-new_action))
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean() #+ (new_next_action-new_next_action).mean()
		# actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean() + action_loss
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

		# --- Actor Update ---
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




		self.total_it += 1


	def save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.log_alpha, filename + '_log_alpha')
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_alpha = torch.load(filename + '_log_alpha')
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
