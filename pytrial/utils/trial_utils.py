'''
Tools to download data and process data from clinicaltrials.gov.
Part of codes come from pytrials: https://pytrials.readthedocs.io/en/latest/index.html.
'''
import pdb
import requests
import csv
import re
import os
import pdb
import time
import zipfile

import pandas as pd
import wget

class ClinicalTrials:
    '''Utilities for download and preprocess trial datasets from ClinicalTrial.gov.
    '''
    __raw_txt_dir__ = './aact-raw'
    _BASE_URL = "https://clinicaltrials.gov/api/"
    _INFO = "info/"
    _QUERY = "query/"
    _JSON = "fmt=json"
    _CSV = "fmt=csv"

    def __init__(self) -> None:
        self.api_info = self._api_info()

    @property
    def study_fields(self):
        '''Get the all possible fields to get from ClinicalTrials.gov.
        '''
        fields_list = json_handler(
            f"{self._BASE_URL}{self._INFO}study_fields_list?{self._JSON}"
        )
        return fields_list["StudyFields"]["Fields"]

    def load_data(self,
        input_dir='./datasets/AACT-ClinicalTrial'
        ):
        '''Load the pre-downloaded and processed csv trial documents.

        Parameters
        ----------

        input_dir: str
            The dir of the input data, should have a csv under named 'clinical_trials.csv';
            Or the path direct to the csv file.
        '''
        if os.path.isdir(input_dir):
            input_dir = os.path.join(input_dir, 'clinical_trials.csv')
        df = pd.read_csv(input_dir, index_col=0)
        return df

    def download(self,
        date='20220501',
        output_dir='./datasets/AACT-ClinicalTrial',
        ):
        '''Download a static copy of all clinical trial documents from clinicaltrials.gov.

        Parameters
        ----------
        date: str
            The date of the database copy.

        fields: list[str]
            A list of fields should be included in the downloaded dataframe.

        output_dir: str
            The output directory of the downloaded data.
        '''
        url = f'https://aact.ctti-clinicaltrials.org/static/exported_files/daily/{date}_pipe-delimited-export.zip'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        print(f'Download all the clinical trial records from {url}, save to {output_dir}.')
        filename = wget.download(url, out=os.path.join(output_dir,'./aact-raw.zip'))
        zipf = zipfile.ZipFile(filename, 'r')
        zipf.extractall(os.path.join(output_dir, self.__raw_txt_dir__))
        zipf.close()
        self.process_studies(input_dir=os.path.join(output_dir, self.__raw_txt_dir__),
            output_dir=output_dir)

    def query_studies(self,
        search_expr,
        fields=['NCTId', 'Condition', 'InterventionName', 'Keyword', 'PrimaryOutcomeMeasure', 'BriefTitle', 'EligibilityCriteria', 'DetailedDescription', 'OverallStatus'],
        max_studies=50,
        fmt='csv'):
        '''Query study content for specified fields from the remote clinicaltrial.gov API.
        Retrieves information from the study fields endpoint, which acquires specified information
        from a large (max 1000) studies. To see a list of all possible fields, check the class'
        study_fields attribute.

        Parameters
        ----------
            search_expr (str): A string containing a search expression as specified by
                `their documentation <https://clinicaltrials.gov/api/gui/ref/syntax#searchExpr>`_.

            fields (list(str)): A list containing the desired information fields.

            max_studies (int): An integer indicating the maximum number of studies to return.
                Defaults to 50.

            fmt (str): A string indicating the output format, csv or json. Defaults to csv.

        Returns
        -------
            Either a dict, if fmt='json', or a list of records (e.g. a list of lists), if fmt='csv.
            Both containing the maximum number of study fields queried using the specified search expression.

        Raises
        ------
            ValueError: The number of studies can only be between 1 and 1000
            ValueError: One of the fields is not valid! Check the study_fields attribute
                for a list of valid ones.
            ValueError: Format argument has to be either 'csv' or 'json'
        '''

        if max_studies > 1000 or max_studies < 1:
            raise ValueError("The number of studies can only be between 1 and 1000")
        elif not set(fields).issubset(self.study_fields):
            raise ValueError(
                "One of the fields is not valid! Check the study_fields attribute for a list of valid ones."
            )
        else:
            concat_fields = ",".join(fields)
            req = f"study_fields?expr={search_expr}&max_rnk={max_studies}&fields={concat_fields}"

            if fmt == "csv":
                url = f"{self._BASE_URL}{self._QUERY}{req}&{self._CSV}"
                fields = csv_handler(url)
                return pd.DataFrame.from_records(fields[1:], columns=fields[0])

            elif fmt == "json":
                url = f"{self._BASE_URL}{self._QUERY}{req}&{self._JSON}"
                return json_handler(url)

            else:
                raise ValueError("Format argument has to be either 'csv' or 'json'")

    def get_study_count(self, search_expr):
        """Returns study count for specified search expression
        Retrieves the count of studies matching the text entered in search_expr.

        Parameters
        ----------
            search_expr (str): A string containing a search expression as specified by
                `their documentation <https://clinicaltrials.gov/api/gui/ref/syntax#searchExpr>`_.

        Returns
        -------
            An integer

        Raises
        ------
            ValueError: The search expression cannot be blank.
        """
        if not set(search_expr):
            raise ValueError("The search expression cannot be blank.")
        else:
            req = f"study_fields?expr={search_expr}&max_rnk=1&fields=NCTId"
            url = f"{self._BASE_URL}{self._QUERY}{req}&{self._JSON}"
            returned_data = json_handler(url)
            study_count = returned_data["StudyFieldsResponse"]["NStudiesFound"]
            return study_count

    def _api_info(self):
        last_updated = json_handler(
            f"{self._BASE_URL}{self._INFO}data_vrs?{self._JSON}"
        )["DataVrs"]
        api_version = json_handler(f"{self._BASE_URL}{self._INFO}api_vrs?{self._JSON}")[
            "APIVrs"
        ]
        return api_version, last_updated

    def process_studies(self,
        input_dir='./datasets/aact-raw',
        output_dir='./datasets'
        ):
        '''Process the raw separate delimited trial documents and combine to a complete csv file.
        '''
        start_time = time.time()
        print('processing description {:.1f} sec'.format(time.time() - start_time))
        df_merge = pd.read_csv(os.path.join(input_dir, 'brief_summaries.txt'), sep='|')
        df_merge = df_merge[['nct_id', 'description']]

        print('processing studies {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
        df = df[['nct_id', 'brief_title']]
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['brief_title'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'title'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['title'].fillna(value='none',inplace=True)

        print('processing interventions {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'interventions.txt'), sep='|')
        df = df[['nct_id','name']]
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['name'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'intervention_name'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['intervention_name'].fillna(value='none',inplace=True)

        print('processing conditions/diseases {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'conditions.txt'), sep='|')
        df = df[['nct_id','name']]
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['name'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'disease'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['disease'].fillna('none', inplace=True)

        print('processing keywords {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'keywords.txt'), sep='|')
        df = df[['nct_id','name']]
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['name'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'keyword'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['keyword'].fillna('none', inplace=True)

        print('processing outcomes {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'design_outcomes.txt'), sep='|')
        df = df[['nct_id','measure',]]
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['measure'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'outcome_measure'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['outcome_measure'].fillna('none', inplace=True)

        print('processing eligbility criteria {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'eligibilities.txt'), sep='|')
        df = df[['nct_id','criteria']]
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['criteria'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'criteria'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['criteria'].fillna('none', inplace=True)

        print('processing references {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir, 'study_references.txt'), sep='|')
        df = df[['nct_id','citation',]]
        df['citation'] = df['citation'].apply(lambda x: x.split('.')).apply(lambda x: x[1] if len(x)>1 else x[0])
        df = df.groupby('nct_id').apply(lambda x: ', '.join(map(str,x['citation'])))
        df = df.reset_index(drop=False)
        df = df.rename(columns={0:'reference'})
        df_merge = pd.merge(df_merge, df, on='nct_id', how='outer')
        df_merge['reference'].fillna('none', inplace=True)
        df_merge.loc[df_merge['reference'] == '']['reference'] = 'none'

        print('processing study status {:.1f} sec'.format(time.time() - start_time))
        df = pd.read_csv(os.path.join(input_dir,'studies.txt'), sep='|')
        df_merge = pd.merge(df_merge, df[['nct_id', 'overall_status']], on='nct_id', how='outer')
        df_merge['overall_status'].fillna('none', inplace=True)
        df_merge.fillna('none', inplace=True)
        if os.path.isdir(output_dir):
            save_path = os.path.join(output_dir, 'clinical_trials.csv')
        else:
            save_path = output_dir
        df_merge = df_merge[df_merge['overall_status'] != 'Withheld'].reset_index(drop=True)
        df_merge.to_csv(save_path)
        print(f'saving processed the csv file to {save_path}')
        return df_merge

    def __repr__(self):
        return f"ClinicalTrials.gov client v{self.api_info[0]}, database last updated {self.api_info[1]}"

def request_ct(url):
    """Performs a get request that provides a (somewhat) useful error message."""
    try:
        response = requests.get(url)
    except ImportError:
        raise ImportError(
            "Couldn't retrieve the data, check your search expression or try again later."
        )
    else:
        return response

def json_handler(url):
    """Returns request in JSON (dict) format"""
    return request_ct(url).json()

def csv_handler(url):
    """Returns request in CSV (list of records) format"""
    response = request_ct(url)
    decoded_content = response.content.decode("utf-8")
    split_by_blank = re.split(r"\n\s*\n", decoded_content)  # Extracts header info
    cr = csv.reader(split_by_blank[1].splitlines(), delimiter=",")
    records = list(cr)
    return records

def test():
    client = ClinicalTrials()
    print(client.study_fields)

    df = client.query_studies(search_expr='Coronavirus+COVID',
        max_studies=500,
        )

    # df = client.process_studies('./datasets/AACT-ClinicalTrial/aact-raw', './datasets/AACT-ClinicalTrial')
    # print(df.head())

if __name__ == '__main__':
    test()
