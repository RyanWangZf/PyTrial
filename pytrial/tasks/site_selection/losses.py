import pdb    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#######
### Loss Functions Components
#######
# ordered_score = (batch, numMC, M)
def compute_log_prob(rank_samples, score, num_MC, K):
	ordered_score = torch.gather(score.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(score.device)
	first_k = ordered_score[:,:,:K]
	permuted_indices = torch.argsort(torch.rand(*first_k.shape), dim=-1).to(score.device)
	permuted_first_k = torch.gather(first_k, -1, permuted_indices)
	random_top_k_ordered_score = torch.cat((permuted_first_k, ordered_score[:,:,K:]), dim=-1)
	denominators = torch.flip(torch.cumsum(torch.flip(random_top_k_ordered_score, [-1]), -1), [-1])
	ranking_prob = torch.prod((random_top_k_ordered_score / denominators)[:,:,:K], -1)	
	prob = math.factorial(K) * ranking_prob
	return torch.log(prob)

def compute_prob(rank_samples, score, num_MC, K):
	ordered_score = torch.gather(score.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(score.device)
	first_k = ordered_score[:,:,:K]
	permuted_indices = torch.argsort(torch.rand(*first_k.shape), dim=-1).to(score.device)
	permuted_first_k = torch.gather(first_k, -1, permuted_indices)
	random_top_k_ordered_score = torch.cat((permuted_first_k, ordered_score[:,:,K:]), dim=-1)
	denominators = torch.flip(torch.cumsum(torch.flip(random_top_k_ordered_score, [-1]), -1), [-1])
	ranking_prob = torch.prod((random_top_k_ordered_score / denominators)[:,:,:K], -1)	
	prob = math.factorial(K) * ranking_prob
	return prob

def ranking_loss(rank_samples, relevance, num_MC, K):
	relevance = relevance/torch.sum(relevance, 1, keepdim=True)
	relevance[relevance != relevance] = 0 # Handle division by zero and set to zero
	relevance = torch.gather(relevance.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(relevance.device)
	return torch.sum(relevance[:,:,:K], 2).to(relevance.device) - torch.sum(relevance[:,:,K:], 2).to(relevance.device)

def fairness_loss(rank_samples, eth_list, relevance, num_MC, K):
	relevance = torch.gather(relevance.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(eth_list.device)[:,:,:K]
	relevance = relevance/torch.sum(relevance, 2, keepdim=True) # Normalize the relevance into weights
	relevance[relevance != relevance] = 1 / K # Handle division by zero and set to even weighting
	eth_list = torch.gather(eth_list.unsqueeze(1).repeat(1,num_MC,1,1), 2, rank_samples.unsqueeze(-1).repeat(1,1,1,eth_list.size(-1)))[:,:,:K,:]
	eth_list = eth_list * relevance.unsqueeze(-1) # Weight the distributions
	delta_f = torch.sum(eth_list, dim=2) # Sum across the chosen K
	reward = Categorical(probs = delta_f).entropy()
	return reward

#######
### Complete Loss Functions
#######

# score = bs * M
# relevance = bs * M
# eth_list = bs * M * num_cat
# rank_samples = bs * MC * M
def loss_function(score, relevance, eth_list, lam, K):	
	num_MC_samples = 25
	M = len(score[0])
	score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score, num_samples=M, replacement=False).to(score.device) for _ in range(num_MC_samples)], axis=1)

	importance_prob_values = compute_log_prob(rank_samples, score, num_MC_samples, K)
	delta_values = ranking_loss(rank_samples, relevance, num_MC_samples, K)
	delta_f_values = fairness_loss(rank_samples, eth_list, relevance, num_MC_samples, K)
	rewards = delta_values + lam*delta_f_values
	loss = -1 * torch.sum(importance_prob_values * rewards, 1)
	return loss/num_MC_samples #, torch.mean(rewards), torch.mean(delta_values), torch.mean(delta_f_values)

def loss_function_enrollment(score, relevance, K):	
	num_MC_samples = 25
	M = len(score[0])
	score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score, num_samples=M, replacement=False).to(score.device) for _ in range(num_MC_samples)], axis=1)

	importance_prob_values = compute_log_prob(rank_samples, score, num_MC_samples, K)
	delta_values = ranking_loss(rank_samples, relevance, num_MC_samples, K)
	rewards = delta_values
	loss = -1 * torch.sum(importance_prob_values * rewards, 1)
	return loss/num_MC_samples #, torch.mean(rewards), torch.mean(delta_values)

def loss_function_fairness(score, relevance, eth_list, K):
	num_MC_samples = 25
	M = len(score[0])
	score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score,num_samples=M, replacement=False).to(score.device) for _ in range(num_MC_samples)], axis=1)
	
	importance_prob_values = compute_log_prob(rank_samples, score, num_MC_samples, K)
	delta_f_values = fairness_loss(rank_samples, eth_list, relevance, num_MC_samples, K)

	loss = -1 * torch.sum(importance_prob_values * (delta_f_values), 1)
	return loss/num_MC_samples

class PolicyGradientLossEnrollment(nn.Module):
    def __init__(self, model, K) -> None:
        super().__init__()
        self.model = model
        self.K = K
        self.loss = loss_function_enrollment

    def forward(self, inputs):
        scores = self.model(inputs)
        loss = self.loss(scores, inputs['label'], self.K)
        return {'loss_value':loss.mean()}
    
class PolicyGradientLossFairness(nn.Module):
    def __init__(self, model, K) -> None:
        super().__init__()
        self.model = model
        self.K = K
        self.loss = loss_function_fairness

    def forward(self, inputs):
        scores = self.model(inputs)
        loss = self.loss(scores, inputs['label'], inputs['eth_label'], self.K)
        return {'loss_value':loss.mean()}
    
class PolicyGradientLossCombined(nn.Module):
    def __init__(self, model, K, lam) -> None:
        super().__init__()
        self.model = model
        self.K = K
        self.lam = lam
        self.loss = loss_function

    def forward(self, inputs):
        scores = self.model(inputs)
        loss = self.loss(scores, inputs['label'], inputs['eth_label'], self.lam, self.K)
        return {'loss_value':loss.mean()}