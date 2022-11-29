###### import ######
import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.info')
RDLogger.DisableLog('rdApp.*')
###### import ######



def plot_hist(prefix_name, prediction, label):
	import seaborn as sns
	import matplotlib.pyplot as plt
	figure_name = prefix_name + "_histogram.png"
	positive_prediction = [prediction[i] for i in range(len(label)) if label[i]==1]
	negative_prediction = [prediction[i] for i in range(len(label)) if label[i]==0]
	save_file_name = "results/" + prefix_name.split('/')[-1] + "_positive_negative.pkl"
	pickle.dump((positive_prediction, negative_prediction), open(save_file_name, 'wb'))
	# sns.distplot(positive_prediction, hist=True,  kde=False, bins=20, color = 'blue', label = 'success')  #### bins = 50 -> 20 
	# sns.distplot(negative_prediction, hist=True,  kde=False, bins=20, color = 'red', label = 'fail')
	# plt.xlabel("predicted success probability", fontsize=24)
	# plt.ylabel("frequencies", fontsize = 25)
	# plt.legend(fontsize = 21)
	# plt.tight_layout()
	# # plt.show()
	# plt.savefig(figure_name)
	return 

def replace_strange_symbol(text):
	for i in "[]'\n/":
		text = text.replace(i,'_')
	return text

#  xml read blog:  https://blog.csdn.net/yiluochenwu/article/details/23515923 
def walkData(root_node, prefix, result_list):
	temp_list =[prefix + '/' + root_node.tag, root_node.text]
	result_list.append(temp_list)
	children_node = root_node.getchildren()
	if len(children_node) == 0:
		return
	for child in children_node:
		walkData(child, prefix = prefix + '/' + root_node.tag, result_list = result_list)


def dynamic_programming(s1, s2):
	arr2d = [[0 for i in s2] for j in s1]
	if s1[0] == s2[0]:
		arr2d[0][0] = 1
	for i in range(1, len(s1)):
		if s1[i]==s2[0]:
			arr2d[i][0] = 1
		else:
			arr2d[i][0] = arr2d[i-1][0] 
	for i in range(1,len(s2)):
		if s2[i]==s1[0]:
			arr2d[0][i] = 1 
		else:
			arr2d[0][i] = arr2d[0][i-1]
	for i in range(1,len(s1)):
		for j in range(1,len(s2)):
			if s1[i] == s2[j]:
				arr2d[i][j] = arr2d[i-1][j-1] + 1 
			else:
				arr2d[i][j] = max(arr2d[i-1][j], arr2d[i][j-1])
	return arr2d[len(s1)-1][len(s2)-1]


def get_path_of_all_xml_file():
	input_file = "./data/all_xml"
	with open(input_file, 'r') as fin:
		lines = fin.readlines()
	input_file_lst = [i.strip() for i in lines]
	return input_file_lst 


def remove_multiple_space(text):
	text = ' '.join(text.split())
	return text 

def nctid_2_xml_file_path(nctid):
	assert len(nctid)==11
	prefix = nctid[:7] + "xxxx"
	datafolder = os.path.join("./ClinicalTrialGov/", prefix, nctid+".xml")
	return datafolder 


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp

def smiles2fp(smiles):
	try:
		mol = Chem.MolFromSmiles(smiles)
		fp = fingerprints_from_mol(mol)
		return fp 
	except:
		return np.zeros((1, 2048), np.int32)

def smiles_lst2fp(smiles_lst):
	fp_lst = [smiles2fp(smiles) for smiles in smiles_lst]
	fp_mat = np.concatenate(fp_lst, 0)
	fp = np.mean(fp_mat,0)
	return fp	


#################  for data loader  #################
def clean_protocol(protocol):
	protocol = protocol.lower()
	protocol_split = protocol.split('\n')
	filter_out_empty_fn = lambda x: len(x.strip())>0
	strip_fn = lambda x:x.strip()
	protocol_split = list(filter(filter_out_empty_fn, protocol_split))	
	protocol_split = list(map(strip_fn, protocol_split))
	return protocol_split 

def split_protocol(protocol):
	protocol_split = clean_protocol(protocol)
	inclusion_idx, exclusion_idx = len(protocol_split), len(protocol_split)	
	for idx, sentence in enumerate(protocol_split):
		if "inclusion" in sentence:
			inclusion_idx = idx
			break
	for idx, sentence in enumerate(protocol_split):
		if "exclusion" in sentence:
			exclusion_idx = idx 
			break 		
	if inclusion_idx + 1 < exclusion_idx + 1 < len(protocol_split):
		inclusion_criteria = protocol_split[inclusion_idx:exclusion_idx]
		exclusion_criteria = protocol_split[exclusion_idx:]
		if not (len(inclusion_criteria) > 0 and len(exclusion_criteria) > 0):
			print(len(inclusion_criteria), len(exclusion_criteria), len(protocol_split))
			exit()
		return inclusion_criteria, exclusion_criteria ## list, list 
	else:
		return protocol_split, 

def protocol2feature(protocol, sentence_2_vec):
	result = split_protocol(protocol)
	inclusion_criteria, exclusion_criteria = result[0], result[-1]
	inclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in inclusion_criteria if sentence in sentence_2_vec]
	exclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in exclusion_criteria if sentence in sentence_2_vec]
	if inclusion_feature == []:
		inclusion_feature = torch.zeros(1,768)
	else:
		inclusion_feature = torch.cat(inclusion_feature, 0)
	if exclusion_feature == []:
		exclusion_feature = torch.zeros(1,768)
	else:
		exclusion_feature = torch.cat(exclusion_feature, 0)
	return inclusion_feature, exclusion_feature

def smiles_txt_to_lst(text):
	"""
		"['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 
		  'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
	"""
	text = text[1:-1]
	lst = [i.strip()[1:-1] for i in text.split(',')]
	return lst 

def icdcode_text_2_lst_of_lst(text):
	text = text[2:-2]
	lst_lst = []
	for i in text.split('", "'):
		i = i[1:-1]
		lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
	return lst_lst

def trial_collate_fn(x):
	sentence2vec = dict()
	nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
	label_vec = default_collate([int(i[1]) for i in x])  ### shape n, 
	smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
	icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
	criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
	return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst]


if __name__ == "__main__":
	text = "interpret_result/NCT00329602__completed____1__1.7650960683822632__phase 4__['restless legs syndrome']__['placebo', 'ropinirole'].png"
	print(replace_strange_symbol(text))






# if __name__ == "__main__":
# 	input_file_lst = get_path_of_all_xml_file() 
# 	print(input_file_lst[:5])
# '''
# input_file_lst = [ 
# 	'ClinicalTrialGov/NCT0000xxxx/NCT00000102.xml', 
#  	'ClinicalTrialGov/NCT0000xxxx/NCT00000104.xml', 
#  	'ClinicalTrialGov/NCT0000xxxx/NCT00000105.xml', 
# 	  ... ]
# '''



# if __name__ == "__main__":
# 	s1 = "328943"
# 	s2 = "13785"
# 	assert dynamic_programming(s1, s2)==2 





