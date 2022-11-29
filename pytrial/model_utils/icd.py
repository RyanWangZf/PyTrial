'''
Provide a series of tools for ICD-9/10 codes.
(1) extract ICD codes given string terms.
(2) get parental and children nodes for ICD-9 codes.
'''
import requests
import os
import json
import wget
import pdb

import pandas as pd
import networkx as nx

NIH_API_PREFIX_ICD10 = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms='
NIH_API_PREFIX_ICD9_DX = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?sf=code,long_name&terms='
NIH_API_PREFIX_ICD9_SG = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_sg/v3/search?sf=code,long_name&terms='
NIH_API_PREFIX_ICD9_CONDITION = 'https://clinicaltables.nlm.nih.gov/api/conditions/v3/search?df=primary_name&terms='
NIH_API_PREFIX_ICD10_CONDITION = 'https://clinicaltables.nlm.nih.gov/api/conditions/v3/search?df=term_icd10cm_codes,primary_name&terms='
ICD9_SG_URL = 'https://github.com/RyanWangZf/PyTrial/raw/main/resources/CMS32_DESC_LONG_SHORT_SG.xlsx'
ICD9_DX_URL = 'https://github.com/RyanWangZf/PyTrial/raw/main/resources/CMS32_DESC_LONG_SHORT_DX.xlsx'
ICD9_GRAPH_URL = 'https://storage.googleapis.com/pytrial/resources/icd-9-hierarchy.json'

def get_icd10_from_nih(term):
    '''
    Query related ICD-10 codes for input terms.

    Parameters
    ----------
    term: str or list[str]
        Disease names or a list of disease names.

    Returns
    -------
    Outputs ICD codes: list[str] or list[list[str]]
    '''
    if isinstance(term, str):
        terms = [term]
    else:
        terms = term

    outputs = []
    for term_ in terms:
        url = NIH_API_PREFIX_ICD10 + term_
        response = requests.get(url)
        text = response.text
        if text == '[0,[],null,[]]':
            outputs.append(None)
            continue
        text = text[1:-1]
        idx1 = text.find('[')
        idx2 = text.find(']')
        codes = text[idx1+1:idx2].split(',')
        codes = [i[1:-1] for i in codes]
        outputs.append(codes)

    if isinstance(term, str):
        outputs = outputs[0]

    return outputs

def get_icd9dx_from_nih(term):
    '''
    Query related ICD-9-CM diagnosis codes for input terms.

    Parameters
    ----------
    term: str or list[str]
        Disease names or a list of disease names.

    Returns
    -------
    Outputs ICD codes: list[str] or list[list[str]]
    '''
    if isinstance(term, str):
        terms = [term]
    else:
        terms = term

    outputs = []
    for term_ in terms:
        url = NIH_API_PREFIX_ICD9_DX + term_
        response = requests.get(url)
        text = response.text
        if text == '[0,[],null,[]]':
            outputs.append(None)
            continue
        text = text[1:-1]
        idx1 = text.find('[')
        idx2 = text.find(']')
        codes = text[idx1+1:idx2].split(',')
        codes = [i[1:-1] for i in codes]
        outputs.append(codes)

    if isinstance(term, str):
        outputs = outputs[0]

    return outputs

def get_icd9sg_from_nih(term):
    '''
    Query related ICD-9-CM procedure codes for input terms.

    Parameters
    ----------
    term: str or list[str]
        Disease names or a list of disease names.

    Returns
    -------
    Outputs ICD codes: list[str] or list[list[str]]
    '''
    if isinstance(term, str):
        terms = [term]
    else:
        terms = term

    outputs = []
    for term_ in terms:
        url = NIH_API_PREFIX_ICD9_SG + term_
        response = requests.get(url)
        text = response.text
        if text == '[0,[],null,[]]':
            outputs.append(None)
            continue
        text = text[1:-1]
        idx1 = text.find('[')
        idx2 = text.find(']')
        codes = text[idx1+1:idx2].split(',')
        codes = [i[1:-1] for i in codes]
        outputs.append(codes)

    if isinstance(term, str):
        outputs = outputs[0]

    return outputs

def get_condition_synonym_from_nih(term):
    '''
    Query relevant medical conditions taking input symptoms/diseases using API: https://clinicaltables.nlm.nih.gov/apidoc/conditions/v3/doc.html

    Parameters
    ----------
    term: str or list[str]
        Disease names or a list of disease names.

    Returns
    -------
    Outputs ICD codes: list[str] or list[list[str]]
    '''
    if isinstance(term, str):
        terms = [term]
    else:
        terms = term

    outputs = []
    for term_ in terms:
        url = NIH_API_PREFIX_ICD9_CONDITION + term_
        response = requests.get(url)
        text = response.text
        if text == '[0,[],null,[]]':
            outputs.append(None)
            continue
        text = text[1:-1]
        idx1 = text.find('[')
        idx2 = text.find(']')
        text = text[idx2+1:]
        idx1 = text.find('[')
        names = text[idx1+1:-1].split(',')
        names = [n[2:-2].lower() for n in names]
        outputs.append(names)

    if isinstance(term, str):
        outputs = outputs[0]

    return outputs

class ICDGraphBase:
    '''
    The base class for ICD9/10 graph.
    '''
    def __init__(self, filename):
        hierarchy = json.loads(open(filename, 'r').read())
        self.graph = nx.readwrite.json_graph.tree_graph(
            hierarchy['tree'],
            attrs={'id':'id', 'children':'children', 'description':'description'}
        )

    def children(self, code):
        '''Return children nodes of code.

        Returns
        -------
        children: list[dict]
            The children codes and their descriptions.
        '''
        code = code.replace('.','')
        node_list = list(self.graph.successors(code))
        return_list = []
        for node in node_list:
            return_list.append(self.__getitem__(node))
        return return_list

    def parent(self, code):
        '''Return the parent node of code.

        Returns
        -------
        parent: dict
            The parent code and its descriptions.
        '''
        code = code.replace('.','')
        node_list = list(self.graph.predecessors(code))
        return_list = []
        for node in node_list:
            return_list.append(self.__getitem__(node))
        return return_list[0]

    def siblings(self, code):
        '''Return sibling nodes of code.

        Returns
        -------
        siblings: list[dict]
            The sibling codes and their descriptions.
        '''
        code = code.replace('.', '')
        parentnode = self.parent(code)['code']
        return self.children(parentnode)

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    @property
    def nxgraph(self):
        return self.graph

    def __getitem__(self, code):
        '''Return the description dict of the codes.
        '''
        code = code.replace('.','')
        return_dict = self.graph.nodes()[code]
        return_dict['code'] = code
        return return_dict

class ICD9Graph(ICDGraphBase):
    '''
    Get an ICD-9 knowledge graph to query parental and children nodes for each code.

    Returns
    -------
    self.graph: nx.DiGraph
        The hierarchy of ICD codes stored as graph in networkx.

    self.codes: list[str]
        All the unique codes.
    '''
    def __init__(self, input_dir=None):
        if input_dir is None:
            input_dir = './resources/icd9'
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            # download the ICD9 hierarchy
            wget.download(ICD9_GRAPH_URL, input_dir)
        
        filename = os.path.join(input_dir, 'icd-9-hierarchy.json')
        super().__init__(filename)


class ICD10Graph(ICDGraphBase):
    '''
    Get an ICD-10 knowledge graph to query parental and children nodes for each code.

    Parameters
    ----------
    input_dir: str
        The dir that stores the hierarchy files.

    version: {'2022', '2021','2020','2019'}
        The version of ICD-10 codes.

    Returns
    -------
    self.graph: nx.DiGraph
        The hierarchy of ICD codes stored as graph in networkx.

    self.codes: list[str]
        All the unique codes.
    '''
    def __init__(self, input_dir=None, version='2021'):

        if input_dir is None:
            input_dir = './resources/icd10'

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            # download the ICD10 hierarchy
            url = f'https://github.com/icd-codex/icd-codex/raw/dev/icdcodex/data/icd-10-{version}-hierarchy.json'
            wget.download(url, input_dir)

        filename = os.path.join(input_dir, f'icd-10-{version}-hierarchy.json')
        super().__init__(filename)

class ICD9_DX_VOC:
    '''
    Get a vocabulary containing the mapping of ICD9 Diagnosis code and its names.

    Parameters
    ----------
    input_dir: str
        The dir that stores ICD9-dx file.    
    '''
    def __init__(self, input_dir='./resources') -> None:
        url = ICD9_DX_URL
        filename = os.path.join(input_dir, 'CMS32_DESC_LONG_SHORT_DX.xlsx')
        if not os.path.exists(filename):
            # download to disk
            wget.download(url, out=filename)
        self.df = pd.read_excel(filename, dtype={'DIAGNOSIS CODE':str})
    
    def __getitem__(self, code):
        index = self.df['DIAGNOSIS CODE'].isin([code])
        sum_index = index.sum()
        if sum_index == 0:
            return None
        else:
            return self.df[index]['LONG DESCRIPTION'].tolist()[0]

    def code2desc(self, code):
        '''
        Get description of codes.

        Parameters
        ----------
        code: str or List[str]
            The input icd code or list of codes.
        '''
        if isinstance(code, str):
            code = [code]
        res = []
        for code_ in code:
            res.append(self.__getitem__(code_))
        if len(res) == 1:
            return res[0]
        else:
            return res

class ICD9_SG_VOC:
    '''
    Get a vocabulary containing the mapping of ICD9 procedure code and its names.

    Parameters
    ----------
    input_dir: str
        The dir that stores ICD9-sg file.
    '''
    def __init__(self, input_dir='./resources') -> None:
        url = ICD9_SG_URL
        filename = os.path.join(input_dir, 'CMS32_DESC_LONG_SHORT_SG.xlsx')
        if not os.path.exists(filename):
            # download to disk
            wget.download(url, out=filename)
        self.df = pd.read_excel(filename, dtype={'PROCEDURE CODE':str})

    def __getitem__(self, code):
        index = self.df['PROCEDURE CODE'].isin([code])
        sum_index = index.sum()
        if sum_index == 0:
            return None
        else:
            return self.df[index]['LONG DESCRIPTION'].tolist()[0]

    def code2desc(self, code):
        '''
        Get description of codes.

        Parameters
        ----------
        code: str or List[str]
            The input icd code or list of codes.
        '''
        if isinstance(code, str):
            code = [code]
        res = []
        for code_ in code:
            res.append(self.__getitem__(code_, 'LONG DESCRIPTION'))
        if len(res) == 1:
            return res[0]
        else:
            return res

if __name__ == '__main__':
    # graph = ICD9Graph('./resources')
    # graph = ICD10Graph('./resources')
    # print(get_icd10_from_nih(["lung neoplasm", "breast"]))
    # print(get_icd9dx_from_nih(["lung neoplasm", "breast"]))
    # print(get_icd9sg_from_nih(["lung neoplasm", "breast"]))
    # print(get_condition_synonym_from_nih(['gastroenteri','salmonella']))
    pass
