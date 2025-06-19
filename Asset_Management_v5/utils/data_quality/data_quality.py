import numpy as np

class DataQuality:
    def __init__(self, logger):
        self.logger = logger

    def get_cases(self, pre_df):
        self.logger.info('Beginning Execution of get_cases method in data_quality.py')
        df = pre_df.copy()
        artifical_indicators_from_wt = [
            'Artificial_Flag_Cost_Health_Asset_Indicator',
            'Artificial_Flag_TMEF_Health_Asset_Indicator',
            'Artificial_Flag_Down_Time_Health_Asset_Indicator',
            'Artificial_Flag_Fails_Index_Health_Asset_Indicator'
        ]
        df['sum_artificial_indicators'] = df[artifical_indicators_from_wt].sum(axis=1)
        df['case_4'] = np.where(df['General_Health_Store_Indicator'].isna(), '4', '')
        df['case_3'] = np.where(df['sum_artificial_indicators'] == len(artifical_indicators_from_wt), '3', '')
        df['case_2'] = np.where(
            (df['sum_artificial_indicators'] < len(artifical_indicators_from_wt)) &
            (df['sum_artificial_indicators'] != 0),
            '2', ''
        )
        df['case_1'] = np.where(
            (df['sum_artificial_indicators'] == 0) &
            (df['General_Health_Store_Indicator'].isna()),
            '1', ''
        )
        df['Case'] = df['case_1'] + df['case_2'] + df['case_3'] + df['case_4']
        df = df.drop(columns=['case_4', 'case_3', 'case_2', 'case_1'])
        self.logger.info('Success Execution of get_cases method in data_quality.py')
        return df
