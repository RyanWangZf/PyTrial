import torch
import numpy as np
import sklearn.metrics
			
def rel_err(score, label, K):
	avg_delta=torch.sum(torch.gather(label, 1, torch.argsort(score, axis=1, descending=True)[:,:K]), dim=1)
	max_delta=torch.sum(torch.sort(label, descending=True, axis=1)[0][:,:K], dim=1)
	return ((max_delta - avg_delta)/max_delta).tolist()

def recall_K(score, label, K):
	model_ind = torch.argsort(score, axis=1, descending=True)[:,:K].detach().cpu().tolist()
	model_ind = [[j for j in model_ind[i] if label[i][j] != 0] for i in range(len(score))] # number of rel. items in the model's top-K

	gt_ind = torch.argsort(label, axis=1, descending=True)[:,:K].detach().cpu().tolist()
	gt_ind = [[j for j in gt_ind[i] if label[i][j] != 0] for i in range(len(label))] # number of actual rel. items

	if gt_ind:
		return [len(set(m_i)&set(g_i))/len(g_i) for m_i, g_i in zip(model_ind, gt_ind) if g_i]
	else:
		return []

def precision_K(score, label, K):
	model_ind = torch.argsort(score, axis=1, descending=True)[:,:K].detach().cpu().tolist()
	model_ind = [[j for j in model_ind[i] if label[i][j] != 0] for i in range(len(score))] # number of rel. items in the model's top-K

	gt_ind = torch.argsort(label, axis=1, descending=True)[:,:K].detach().cpu().tolist()
	gt_ind = [[j for j in gt_ind[i] if label[i][j] != 0] for i in range(len(label))] # number of actual rel. items

	if gt_ind:
		return [len(set(m_i)&set(g_i))/len(m_i) for m_i, g_i in zip(model_ind, gt_ind) if m_i]
	else:
		return []

def NDCG_K(score, label, K):
	nz_scores = [[score[i][j].item() for j in range(len(label[i]))] for i in range(len(score))]
	nz_labels = [[label[i][j].item() for j in range(len(label[i]))] for i in range(len(label))]
	
	if nz_labels:
		if len(nz_labels[0]) >= K:
			return [sklearn.metrics.ndcg_score(np.asarray([nz_labels[i]]), np.asarray([nz_scores[i]]), k=K) for i in range(len(score))]
		else:
			try:
				return [sklearn.metrics.ndcg_score(np.asarray([nz_labels[i]]), np.asarray([nz_scores[i]])) for i in range(len(score))]
			except: 
				return []

def compute_repn(score, label, K, eth_sel):
	net_inds = torch.argsort(score, axis=1, descending=True)[:,:K].squeeze(-1)
	base_inds = torch.argsort(label, axis=1, descending=True)[:,:K].squeeze(-1)
	net_weights = torch.gather(label, 1, net_inds)
	base_weights = torch.gather(label, 1, base_inds)
	net_weights = net_weights/torch.sum(net_weights, 1, keepdim=True)
	base_weights = base_weights/torch.sum(base_weights, 1, keepdim=True)
	net_weights[net_weights != net_weights] = 1 / K # Handle division by zero and set to even weighting
	base_weights[base_weights != base_weights] = 1 / K # Handle division by zero and set to even weighting
	net_eth = torch.gather(eth_sel, 1, net_inds.unsqueeze(-1).repeat(1,1,eth_sel.size(-1))) * net_weights.unsqueeze(-1) / 100
	base_eth = torch.gather(eth_sel, 1, base_inds.unsqueeze(-1).repeat(1,1,eth_sel.size(-1))) * base_weights.unsqueeze(-1) / 100
	net_repn = torch.sum(net_eth, dim=1)
	base_repn = torch.sum(base_eth, dim=1)
	return net_repn, base_repn