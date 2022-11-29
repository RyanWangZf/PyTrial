'''
Provide a series of tools for getting drug mappings, e.g., name to SMILES string,
drug-DDI matrix, drug ndc - rxcui - atc4 codes, drug cid-atc codes.
'''
import json
import wget
import os
import requests
import re
import pdb
import zipfile
import gzip

import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
from tqdm import tqdm

from ..utils.tabular_utils import read_csv_to_df, read_txt_to_df, read_excel_to_df
from ..utils.check import make_dir_if_not_exist

# DRUG_DDI_URL = 'https://uofi.box.com/shared/static/xdfgrnhzotz6ktyrdsrikrnz1th97fic.csv'
DRUG_DDI_URL = 'https://storage.googleapis.com/pytrial/drug-DDI.csv'
# DRUG_BANK_URL = 'https://uofi.box.com/shared/static/4f3g4dvdfyz5goubazeqfzdi0abcgzwf.csv'
DRUG_BANK_URL = 'https://storage.googleapis.com/pytrial/drugbank_drugs_info.csv'
RXCUI_ATC4_NDC11_URL = 'https://github.com/RyanWangZf/PyTrial/raw/main/resources/rxcui_atc4_ndc11.zip'
NDC_NAME_URL = 'https://github.com/RyanWangZf/PyTrial/raw/main/resources/ndc_name.csv'
NAME_SMILES_URL = 'https://github.com/RyanWangZf/PyTrial/raw/main/resources/drug_smiles.csv'
NAME_ATC_URL ='https://github.com/RyanWangZf/PyTrial/raw/main/resources/atc_drug.csv'
# ATC5_NDC_URL = 'https://uofi.box.com/shared/static/dk07wip4l4hkbolp09e3rz3un4hokocb.zip'
ATC5_NDC_URL = 'https://storage.googleapis.com/pytrial/atc5_ndc.zip'
# ATC_DEF_URL = 'https://uofi.box.com/shared/static/tdz6glo9waf353mwxvqw44l6r43vrfqm.zip'
ATC_DEF_URL = 'https://storage.googleapis.com/pytrial/ATC.csv.zip'
# FDA_NDC_NAME_URL = 'https://uofi.box.com/shared/static/ah6gk3ljaj0uz0cr2yoecmd3so2onih7.zip'
FDA_NDC_NAME_URL = 'https://storage.googleapis.com/pytrial/fda_ndc.zip'

class DrugTransformer:
    '''
    Provide a series of drug-related functions for

    (1) drug name to atc / atc to drug name
    (2) drug name to ndc-11 / ndc-11 to drug name
    (3) drug name to smiles (molecule structure)
    (4) atc to smiles
    (5) ndc-11 to smiles
    (6) ndc-11 to atc / atc to ndc

    To convert ndc-10 ditis to ndc-11, use `convert_ndc10_ndc11`.
    '''
    def __init__(self):
        # initialize using several preprocessed files
        self._build_ndc_rxcui_atc_map()
        self._build_ndc_name_map()
        self._build_name_smiles()
        self._build_atc_name()
        self._build_atc_ndc_smiles()

    def ndc2atc(self, code):
        '''
        Parameters
        ----------
        code: str or list[str]
            NDC-11 codes.

        Returns
        -------
        atc codes: list[str] or list[list[str,None]]
        '''
        df = self.rxcui_atc4_ndc11
        single_input = isinstance(code, str)
        if single_input: code = [code]
        outputs = []
        for c in code:
            df_sub = df[df['ndc11']==c]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['atc4'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def atc2ndc(self, code):
        '''
        Parameters
        ----------
        code: str or list[str]
            ATC-4 codes.

        Returns
        -------
        NDC-11 codes: list[str] or list[list[str,None]]
        '''
        df = self.rxcui_atc4_ndc11
        single_input = isinstance(code, str)
        if single_input: code = [code]
        outputs = []
        for c in code:
            df_sub = df[df['atc4']==c]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['ndc11'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def name2ndc(self, name):
        '''
        Parameters
        ----------
        name: str or list[str]
            Drug names.

        Returns
        -------
        ndc codes: list[str] or list[list[str,None]]
        '''
        df = self.ndc11_name
        single_input = isinstance(name, str)
        if single_input: name = [name]
        outputs = []
        for c in name:
            df_sub = df[df['drug']==c.lower()]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['ndc11'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def ndc2name(self, code):
        '''
        Parameters
        ----------
        code: str or list[str]
            NDC-11 codes.

        Returns
        -------
        name: list[str] or list[list[str,None]]
            Drug names.
        '''
        df = self.ndc11_name
        single_input = isinstance(code, str)
        if single_input: code = [code]
        outputs = []
        for c in code:
            df_sub = df[df['ndc11']==c.lower()]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['drug'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def atc2name(self, code):
        '''
        Parameters
        ----------
        code: str or list[str]
            ATC4 codes.

        Returns
        -------
        names: list[str] or list[list[str,None]]
            Drug names.
        '''
        df = self.name_atc
        single_input = isinstance(code, str)
        if single_input: code = [code]
        outputs = []
        for c in code:
            df_sub = df[df['atc4']==c]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['drug'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def name2atc(self, name):
        '''
        Parameters
        ----------
        name: str or list[str]
            Drug names.

        Returns
        -------
        codes: list[str] or list[list[str,None]]
            ATC4 codes.
        '''
        df = self.name_atc
        single_input = isinstance(name, str)
        if single_input: name = [name]
        outputs = []
        for c in name:
            df_sub = df[df['drug']==c.lower()]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['atc4'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def name2smiles(self, name):
        '''
        Parameters
        ----------
        name: str or list[str]
            Drug names.

        Returns
        -------
        smiles: list[str] or list[list[str,None]]
            Drug molecule structures represented by SMILES.
        '''
        df = self.name_smiles
        single_input = isinstance(name, str)
        if single_input: name = [name]
        outputs = []
        for c in name:
            df_sub = df[df['drug']==c.lower()]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['moldb_smiles'].tolist())

        if single_input: return outputs[0]
        else: return outputs

    def atc2smiles(self, code):
        '''
        Parameters
        ----------
        code: str or list[str]
            ATC4 codes.

        Returns
        -------
        smiles: list[str] or list[list[str,None]]
            Drug molecule structures represented by SMILES.
        '''
        df = self.atc_smiles
        single_input = isinstance(code, str)
        if single_input: code = [code]
        outputs = []
        for c in code:
            df_sub = df[df['atc4']==c]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['moldb_smiles'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def ndc2smiles(self, code):
        '''
        Parameters
        ----------
        name: str or list[str]
            Drug names.

        Returns
        -------
        smiles: list[str] or list[list[str,None]]
            Drug molecule structures represented by SMILES.
        '''
        '''
        Parameters
        ----------
        code: str or list[str]
            NDC-11 codes.

        Returns
        -------
        smiles: list[str] or list[list[str,None]]
            Drug molecule structures represented by SMILES.
        '''
        df = self.ndc_smiles
        single_input = isinstance(code, str)
        if single_input: code = [code]
        outputs = []
        for c in code:
            df_sub = df[df['ndc11']==c]
            if len(df_sub) == 0:
                outputs.append(None)
            else:
                outputs.append(df_sub['moldb_smiles'].tolist())
        if single_input: return outputs[0]
        else: return outputs

    def _build_ndc_rxcui_atc_map(self):
        # load ndc_rxcui_atc map csv
        filename = './resources/rxcui_atc4_ndc11.csv'
        make_dir_if_not_exist('resources')
        if not os.path.exists('./resources/rxcui_atc4_ndc11.csv'):
            wget.download(RXCUI_ATC4_NDC11_URL, out=filename.replace('.csv','.zip'))
            f = zipfile.ZipFile(filename.replace('.csv','.zip'),'r')
            f.extractall('./resources')
            f.close()
            print(f'Download RXCUI-NDC-ATC4 mapping file to {filename}.')
        self.rxcui_atc4_ndc11 = read_csv_to_df(filename, dtype={'ndc11':str, 'rxcui':str}, low_memory=False)

    def _build_ndc_name_map(self):
        filename = './resources/ndc_name.csv'
        make_dir_if_not_exist('resources')
        if not os.path.exists(filename):
            wget.download(NDC_NAME_URL, out=filename)
            print(f'Download NDC-DrugName mapping file to {filename}.')
        self.ndc11_name = read_csv_to_df(filename, dtype={'ndc11':str, 'DRUG':str}, low_memory=False)
        self.ndc11_name = self.ndc11_name.applymap(lambda x: x.lower())

    def _build_name_smiles(self):
        filename = './resources/drug_smiles.csv'
        make_dir_if_not_exist('resources')
        if not os.path.exists(filename):
            wget.download(NAME_SMILES_URL, out=filename)
            print(f'Download DrugName-SMILES mapping file to {filename}.')
        self.name_smiles = read_csv_to_df(filename, dtype={'DRUG':str, 'moldb_smiles':str}, low_memory=False)
        self.name_smiles['drug'] = self.name_smiles['drug'].apply(lambda x: x.lower())
        self.name_smiles = self.name_smiles.dropna()

    def _build_atc_name(self):
        filename = './resources/atc_drug.csv'
        make_dir_if_not_exist('resources')
        if not os.path.exists(filename):
            wget.download(NAME_ATC_URL, out=filename)
            print(f'Download DrugName-ATC4 mapping file to {filename}.')
        self.name_atc = read_csv_to_df(filename, dtype={'DRUG':str, 'atc4':str}, low_memory=False)
        self.name_atc['drug'] = self.name_atc['drug'].apply(lambda x: x.lower())

    def _build_atc_ndc_smiles(self):
        df_name_atc = self.name_atc
        df_name_smiles = self.name_smiles
        df = df_name_atc.merge(df_name_smiles, on='drug')[['atc4','moldb_smiles']]
        self.atc_smiles = df.drop_duplicates().reset_index(drop=True)
        df_ndc_name = self.ndc11_name
        df = df_ndc_name.merge(df_name_smiles, on='drug').drop_duplicates().reset_index(drop=True)
        self.ndc_smiles = df[['ndc11','moldb_smiles']]


def convert_ndc10_ndc11(code):
    '''
    Covert NDC 10 digits with hyphens to NDC 11 digits w/o hyphens.

    Examples
    --------

    10-Digit Format: https://health.maryland.gov/phpa/OIDEOR/IMMUN/Shared%20Documents/Handout%203%20-%20NDC%20conversion%20to%2011%20digits.pdf
    on Drug Package	10-Digit Format Example	11-Digit Format	11-Digit Format Example	10-Digit NDC Example	11-Digit Conversion of 10-Digit NDC Example
    4-4-2	9999-9999-99	5-4-2	09999-9999-99	0002-7597-01	00002-7597-01
    5-3-2	99999-999-99	5-4-2	99999-0999-99	50242-040-62	50242-0040-62
    5-4-1	99999-9999-9	5-4-2	99999-9999-09	60574-4114-1	60574-4114-01
    '''
    s = code.split('-')
    S = []
    for i, L in enumerate([5,4,2]):

        if len(s[i]) < L:
            S.append('0'*(L-len(s[i])) + s[i])
        else:
            S.append(s[i])

    return ''.join(S)


def download_drug_ddi(output_dir='./datasets'):
    '''
    Download the drug-ddi information file to disk.

    Parameters
    ----------
    output_dir: str
        The output dir.
    '''
    make_dir_if_not_exist(output_dir)
    filename = os.path.join(output_dir, 'drug-DDI.csv')
    wget.download(DRUG_DDI_URL, out=filename)
    print(f'Save drug DDI file to {filename}.')


def download_drug_bank(output_dir='./datasets'):
    '''
    Download the drug-bank information file to disk.
    Contains drugnames and drug SMILES molecule strings.

    Parameters
    ----------
    output_dir: str
        The output dir.
    '''
    make_dir_if_not_exist(output_dir)
    filename = os.path.join(output_dir, 'drugbank.csv')
    wget.download(DRUG_BANK_URL, out=filename)
    print(f'Save drug bank file to {filename}.')

class DrugGraph:
    '''
    Provide tools to get hierarchy of drug by ACT codes.

    From ATC2 - ATC4 (dont include ATC5)
    '''
    def __init__(self, input_dir='./resources'):
        filename = os.path.join(input_dir, 'drug_hierarchy.json')
        if not os.path.exists(filename):
            self.preprocess(input_dir)

        with open(filename, 'r') as f:
            hierarchy = json.loads(f.read())
        self.graph = json_graph.adjacency_graph(hierarchy)
        print('load drug graph from', filename)

    def preprocess(self, dir='./resources'):
        '''
        Process raw data and deposit the graph data to the local disk.
        Search `ndc_atc.csv` under the given directory `dir`. If not,
        download the raw files to the disk and unzip.
        '''
        make_dir_if_not_exist(dir)

        atctree_file = os.path.join(dir, 'ndc_atc.csv')
        if not os.path.exists(atctree_file):
            self._download_ndc_atc_map(dir, atctree_file)

        atcdef_file = os.path.join(dir, 'atc_def.csv')
        if not os.path.exists(atcdef_file):
            self._download_atc_def(dir, atcdef_file)

        fdandc_file = os.path.join(dir, 'fda_ndc_name.csv')
        if not os.path.exists(fdandc_file):
            self._download_fda_ndc_name(dir, fdandc_file)

        df = read_csv_to_df(atctree_file)
        df['ndc'] = df['ndc'].apply(lambda x: convert_ndc10_ndc11(x))
        df = df.dropna(subset=['atc4']).reset_index(drop=True)
        df_atcdef = read_csv_to_df(atcdef_file)
        df_atcdef['atc'] = df_atcdef['class id'].apply(lambda x: x.split('/')[-1])

        # TODO: attach ndc code to its name
        df_ndc = read_csv_to_df(fdandc_file)
        df_ndc['ndc'] = df_ndc['ndcpackagecode'].apply(lambda x: convert_ndc10_ndc11(x))
        df_ndc = df_ndc[['proprietaryname','ndc']]
        nodes_list = []
        for index, row in df_ndc.iterrows():
            nodes_list.append(
                (row['ndc'], {'description':row['proprietaryname']})
            )

        # add ndc code to the leaf code of G
        G = nx.DiGraph()
        G.add_nodes_from(nodes_list)

        atc4 = df['atc4'].unique()
        for code in tqdm(atc4):
            # some drug dont have atc5 code
            ndc_codes = df[df['atc4'] == code]['ndc'].unique()
            edges = list(zip([code]*len(ndc_codes), ndc_codes))
            G.add_edges_from(edges)

        # build atc code graph
        G_atc = self._build_atc_tree(df_atcdef)
        G = nx.compose(G, G_atc)
        res = nx.adjacency_data(G, attrs={'key':'description', 'id':'id'})

        out_filename = os.path.join(dir, 'drug_hierarchy.json')
        with open(out_filename, 'w') as f:
            f.write(json.dumps(res))
        print('done, save the drug hierarchy to', out_filename)

    def _download_fda_ndc_name(self, dir, fdandc_file):
        temp_filename = os.path.join(dir, 'fda_ndc.zip')
        if os.path.exists(temp_filename): os.remove(temp_filename)
        wget.download(FDA_NDC_NAME_URL, out=temp_filename)
        f = zipfile.ZipFile(temp_filename,'r')
        f.extractall(dir)
        f.close()
        temp_textname = os.path.join(dir, 'package.txt')
        temp_textname2 = os.path.join(dir, 'product.txt')
        df_pack = read_txt_to_df(temp_textname, encoding='cp1252')
        df_prod = read_txt_to_df(temp_textname2, encoding='cp1252')
        df = df_pack.merge(df_prod, on='productid')
        df.to_csv(fdandc_file, index=False)
        os.remove(temp_filename)
        os.remove(temp_textname)
        os.remove(temp_textname2)
        print("\n Download raw file to", fdandc_file)

    def _download_ndc_atc_map(self, dir, atctree_file):
        temp_filename = os.path.join(dir, 'ndc_atc.zip')
        if os.path.exists(temp_filename): os.remove(temp_filename)
        wget.download(ATC5_NDC_URL, out=temp_filename)
        f = zipfile.ZipFile(temp_filename,'r')
        f.extractall(dir)
        f.close()
        temp_csvfilename = os.path.join(dir, 'ndc_map 2020_06_17 (atc5 atc4 ingredients).csv')
        os.rename(temp_csvfilename, atctree_file)
        os.remove(temp_filename)
        print("Download raw file to", atctree_file)

    def _download_atc_def(self, dir, atcdef_file):
        temp_filename = os.path.join(dir, 'atc_def.csv.zip')
        if os.path.exists(temp_filename): os.remove(temp_filename)
        wget.download(ATC_DEF_URL, out=temp_filename)
        f = zipfile.ZipFile(temp_filename,'r')
        f.extractall(dir)
        f.close()
        temp_csvfilename = os.path.join(dir, 'ATC.csv')
        os.rename(temp_csvfilename, atcdef_file)
        os.remove(temp_filename)
        print("\n Download raw file to", atcdef_file)

    def _build_atc_tree(self, df_atcdef):
        G = nx.DiGraph()
        df_atc_label = df_atcdef[['atc','preferred label']]
        df_atc_label = df_atc_label[df_atc_label['atc'].map(len) < 7]
        nodes_list = []
        for index, row in df_atc_label.iterrows():
            nodes_list.append(
                (row['atc'], {'description':row['preferred label']})
            )
        nodes_list.append(('root', {'description':'Anatomical Therapeutic Chemical (ATC) Classification'}))
        G.add_nodes_from(nodes_list)
        codes = df_atcdef['atc'].unique()
        codes = pd.Series(codes)
        atc4 = codes[codes.map(len)==5]
        atc3 = codes[codes.map(len)==4]
        atc2 = codes[codes.map(len)==3]
        atc1 = codes[codes.map(len)==1]
        for _, code in tqdm(atc3.iteritems()):
            children = atc4[atc4.apply(lambda x: code == x[:4])]
            edges = list(zip([code]*len(children), children.tolist()))
            G.add_edges_from(edges)
        for _, code in tqdm(atc2.iteritems()):
            children = atc3[atc3.apply(lambda x: code == x[:3])]
            edges = list(zip([code]*len(children), children.tolist()))
            G.add_edges_from(edges)
        for _, code in tqdm(atc1.iteritems()):
            children = atc2[atc2.apply(lambda x: code == x[:1])]
            edges = list(zip([code]*len(children), children.tolist()))
            G.add_edges_from(edges)

        edges = list(zip(['root']*len(atc1), atc1))
        G.add_edges_from(edges)
        return G

# if __name__ == '__main__':
    # download_drug_ddi()
    # download_drug_bank()
