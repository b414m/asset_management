import pandas as pd
import numpy as np
import math
from datetime import date
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from itertools import product
#groupsTMEF = [['ID del activo'], ['# de Tienda'], ['Grupo_Nombre','Formato de negocio'], ['Grupo_Nombre','Formato de negocio','Estado/Provincia']]
from functools import reduce

class HealthIndicators:
    """__summary__"""
    def __init__(self, logger):
        self.logger = logger

    def get_indicators(self, pre_df, strategy_deltas):
        self.logger.info('Beginning Execution of get_indicators method in health_indicators.py')
        df = pre_df.copy()
        last_months = [0, 1, 2, 3, 6, 12, 15, 18, 24]
        group = 'ID del activo'
        target_date_column = 'Termino Real'
        df = self.get_raw_indicators(df, group, strategy_deltas, target_date_column, last_months)
        self.logger.info('Success Execution of get_indicators method in health_indicators.py')
        return df

    def get_indicators_analysis(self, pre_df):
        self.logger.info('Beginning Execution of get_indicators_analysis method in health_indicators.py')
        df = pre_df.copy()
        groups = ['ID del activo']
        indicators = {
            'sum_Costo Total de Proveedor': True,
            'TMEF_Termino Real': False,
            'sum_Tot Current Working Hours': True,
            'Fails_Frequency_Index': True,
            'years_since_grand_opening_x': True,
            'years_since_last_touch_x': True
        }
        # indicators_weights = {
        #     'sum_Costo Total de Proveedor': 0.1,
        #     'TMEF_Termino Real': 0.35,
        #     'sum_Tot Current Working Hours': 0.05,
        #     'Fails_Frequency_Index': 0.1,
        #     'years_since_grand_opening_x': 0.2,
        #     'years_since_last_touch_x': 0.2
        # }
        indicators_weights = {
            'sum_Costo Total de Proveedor': 0.4,
            'TMEF_Termino Real': 0.3,
            'sum_Tot Current Working Hours': 0.025,
            'Fails_Frequency_Index': 0.175,
            'years_since_grand_opening_x': 0.05,
            'years_since_last_touch_x': 0.05
        }
        df = self.map_indicator_using_min_max_and_optimal_value_approach(df, groups, indicators)
        df = self.build_metric(df, groups, indicators_weights)
        self.logger.info('Success Execution of get_indicators_analysis method in health_indicators.py')
        return df
    
    def get_general_services_health(self, pre_df, health_asset_column):
        self.logger.info('Beginning Execution of get_general_services_health method in health_indicators.py')
        df = pre_df.copy()
        df = self.get_general_services_health_by_group(df, '', '# de Tienda', health_asset_column, True)
        df = self.get_general_services_health_by_group(df, '', 'Estado/Provincia', health_asset_column, True)
        self.logger.info('Beginning Execution of get_general_services_health method in health_indicators.py')
        return df
    
    def get_general_services_healthv2(self, pre_df, health_column, services_weights, group):
        self.logger.info('Beginning Execution of get_general_services_healthv2 method in health_indicators.py')
        df = pre_df.copy()
        df['weighted_'+health_column] = 0
        for service, weight in services_weights.items():
            df['weighted_'+health_column] = np.where(
                df['Service']==service,
                df[health_column]*weight,
                df['weighted_'+health_column]
            )
        df['general_services_'+health_column] = df.groupby([group,'Asset ID'])['weighted_'+health_column].transform('sum')
        self.logger.info('Success Execution of get_general_services_healthv2 method in health_indicators.py')
        return df
    
    def removing_outliers(self, pre_df, group, indicator):
        self.logger.info('Beginning Execution of removing_outliers method in health_indicators.py')
        df = pre_df.copy()
        Q1 = df.groupby(group)[indicator+'_'+group].quantile(0.25)
        Q3 = df.groupby(group)[indicator+'_'+group].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['lt_lower_bound'+indicator+'_'+group] = df.groupby(group)[indicator+'_'+group].transform(lambda x: x < lower_bound)
        df['gt_upper_bound'+indicator+'_'+group] = df.groupby(group)[indicator+'_'+group].transform(lambda x: x > upper_bound)
        df['is_outlier'+indicator+'_'+group] = df['lt_lower_bound'+indicator+'_'+group] | df['gt_upper_bound'+indicator+'_'+group]
        df = df[df['is_outlier'+indicator+'_'+group]==True]
        self.logger.info('Success Execution of removing_outliers method in health_indicators.py')
        return df
    
    def remove_outliers(self, pre_df, column):
        self.logger.info('Beginning Execution of remove_outliers in health_indicators.py')
        df = pre_df.copy()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['lt_lower_bound_' + column] = df[column] < lower_bound
        df['gt_upper_bound_' + column] = df[column] > upper_bound
        df['is_outlier_' + column] = (
            df['lt_lower_bound_' + column] |
            df['gt_upper_bound_' + column]
        )
        df = df[df['is_outlier_' + column] == False]
        self.logger.info('Success Execution of remove_outliers in health_indicators.py')
        return df
    
    def mark_outliers(self, group):
        # Calcular el rango intercuartil
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1
        # Identificar outliers
        outliers = (group < (Q1 - 1.5 * IQR)) | (group > (Q3 + 1.5 * IQR))
        return outliers
    
    def map_indicator_to_health_using_logistic_function(
        self,
        pre_df: pd.DataFrame,
        group: str,
        indicator: str,
        strategy: str,
        inv_proportional: bool,
        target_name=None
    ) -> pd.DataFrame:
        self.logger.info(
            'Beginning Execution of map_indicator_to_health_using_logistic_function in health_indicators.py'
        )
        df = pre_df.copy()
        bins = [0, 2, 4, 6, 8, 10]
        labels = ['Crítico Alto', 'Crítico Medio', 'Crítico Bajo', 'Advertencia', 'Correcto']
        groupStr = '_'.join([x.replace('#', '').replace(' ', '') for x in group])
        new_indicator = (indicator + '_' + group).replace('_IDdelactivo_IDdelactivo', '_IDdelactivo')
        if group == '':
            new_indicator = indicator
    
        if strategy == 'standard':
            df['mean_' + new_indicator] = df[new_indicator].mean()
            df['median_' + new_indicator] = df[new_indicator].median()
            df['std_' + new_indicator] = df[new_indicator].std()
            df['normalized_' + new_indicator] = (
                df[new_indicator] - df['mean_' + new_indicator]
            ) / df['std_' + new_indicator]
            Q1 = df[new_indicator].quantile(0.25)
            Q3 = df[new_indicator].quantile(0.75)
            IQR = Q3 - Q1
        
    elif strategy == 'min-max':
        df['min_' + new_indicator] = df[new_indicator].min()
        df['max_' + new_indicator] = df[new_indicator].max()
        df['normalized_' + new_indicator] = (
            df[new_indicator] - df['min_' + new_indicator]
        ) / (
            df['max_' + new_indicator] - df['min_' + new_indicator]
        )
    if inv_proportional == True:
        df['a_' + new_indicator] = 6.9315
    elif inv_proportional == False:
        df['a_' + new_indicator] = -6.9315
    df['k_' + new_indicator] = df['a_' + new_indicator] / IQR
    if target_name == None:
        df['Health_Indicator_Based_On_Logistic_Of_' + new_indicator] = 10 / (
            1 + np.exp(
                df['k_' + new_indicator] * (df[new_indicator] - df['median_' + new_indicator])
            )
        )
        df['Category_Health_Indicator_Based_On_Logistic_Of_' + new_indicator] = pd.cut(
            df['Health_Indicator_Based_On_Logistic_Of_' + new_indicator],
            bins=bins,
            labels=labels,
            right=True
        )
    else:
        df[target_name + '_Indicator'] = 10 / (
            1 + np.exp(
                df['k_' + new_indicator] * (df[new_indicator] - df['median_' + new_indicator])
            )
        )
        df[target_name + '_Category'] = pd.cut(
            df[target_name + '_Indicator'],
            bins=bins,
            labels=labels,
            right=True
        )
        #df['Category_Health_Indicator_Based_On_Logistic_Of_' + new_indicator] = df['Health_Indicator_Based_On_Logistic_Of_' + new_indicator]
        df = df.drop(columns=[
            'mean_' + new_indicator,
            'median_' + new_indicator,
            'std_' + new_indicator,
            'normalized_' + new_indicator,
            'a_' + new_indicator,
            'k_' + new_indicator
        ])
    self.logger.info(
        'Success Execution of map_indicator_to_health_using_logistic_function in health_indicators.py'
    )
    return df

    def get_ots_analysis_cumulated_approach(self, pre_df, last_months):
        self.logger.info(
            'Beginning Execution of get_ots_analysis_cumulated_approach in health_indicators.py'
        )
        df = pre_df.copy()
        values = np.array(last_months)
        df['Year_Month'] = df['Termino Real'].dt.to_period('M')
        weights = 1 / (1 + values)
        normalized_weights = weights / weights.sum()
        weights_column_names = []
        result = pd.DataFrame()
        scaler_global = MinMaxScaler()
        i = 0
        for offset, weight in zip(last_months, normalized_weights):
            current_period = pd.Timestamp.now().to_period('M')
            window_start = current_period - offset
            window_end = (
                current_period - (last_months[i - 1] + 1)
                if i > 0
                else None
            )
            if window_end:
                cumulated_filter = df[
                    (df['Termino Real'] >= window_start.start_time) &
                    (df['Termino Real'] <= window_end.end_time)
                ]
            else:
                cumulated_filter = df[
                    (df['Termino Real'] >= window_start.start_time) &
                    (df['Termino Real'] <= current_period.end_time)
                ]
            if offset == 0:
                column_name = 'WO_since_' + current_period.strftime('%B_%Y')
                weight_column_name = 'weighted_' + column_name
            else:
                column_name = "'WO_since_{0}'".format(window_start.strftime('%B_%Y'))
                weight_column_name = 'weighted_' + column_name

            if cumulated_filter.empty:
                count = df[['ID del activo']].drop_duplicates()
                count[column_name] = 0
            else:
                count = cumulated_filter.groupby('ID del activo')['ID de Tarea'].count().reset_index()
                count.rename(columns={'ID de Tarea': column_name}, inplace=True)

            count[weight_column_name] = count[column_name] * weight
            weights_column_names.append(weight_column_name)

            result = pd.merge(result, count, on='ID del activo', how='outer') if not result.empty else count
            result = result.fillna(0)
            i += 1

        result['Fails_Frequency_Index_ID del activo'] = result[weights_column_names].sum(axis=1)
        report = df[['ID del activo', 'Store_Asset_ID']].merge(result, on='ID del activo', how='left')
        report = report.fillna(0)
        report['Health_Indicator_Based_On_Logistic_Of_Fails_Frequency_Index_ID del activo'] = (
            (1 - scaler_global.fit_transform(
                report[['Fails_Frequency_Index_ID del activo']]
            )) * 10
        )
        # result[['Fails_Frequency_Index_ID del activo']])*10)
        report = report[[
            'ID del activo',
            'Fails_Frequency_Index_ID del activo',
            'Health_Indicator_Based_On_Logistic_Of_Fails_Frequency_Index_ID del activo'
        ]]
        report = report.drop_duplicates(subset=['ID del activo'])
        self.logger.info('Sucess Execution of get_ots_analysis_cumulated_approach in health_indicators.py')
        return report

    def preprocess_indicator(self, pre_df, indicator, group):
        self.logger.info('Beginning Execution of preprocess_indicator in health_indicators.py')
        df = pre_df.copy()
        diferenciator_value = -100
        df[indicator + '_' + group] = df[indicator + '_' + group].fillna(diferenciator_value)
        df['Health_Indicator_Based_On_Logistic_Of_' + indicator + '_' + group] = df['Health_Indicator_Based_On_Logistic_Of_' + indicator + '_' + group].fillna(10)
        df['Artificial_' + indicator + '_' + group] = np.where(df[indicator + '_' + group] == diferenciator_value, True, False)
        df['Comments_' + indicator + '_' + group] = np.where(
            df[indicator + '_' + group] == diferenciator_value,
            indicator + '_' + group + ' value was generated artificially',
            ''
        )
        self.logger.info('Success Execution of preprocess_indicator in health_indicators.py')
        return df

    def map_indicator_using_min_max_and_optimal_value_approach(self, pre_df: pd.DataFrame, groups: list, indicators: dict):
        self.logger.info('Beginning Execution of map_indicator_using_min_max_and_optimal_value_approach in health_indicators.py')
        group = groups[0]
        df = pre_df.reset_index(drop=True).copy()

        indicators_and_fields_implied_relations = {
            'sum_Costo Total de Proveedor': ['ID del activo', '# de Tienda', 'Estado/Provincia', 'Costo Total de Proveedor'],
            'TMEF_Termino Real':            ['ID del activo', '# de Tienda', 'Estado/Provincia', 'Termino Real'],
            'sum_Tot Current Working Hours':['ID del activo', '# de Tienda', 'Estado/Provincia', 'Tot Current Working Hours'],
            'Fails_Frequency_Index':        ['ID del activo', '# de Tienda', 'Estado/Provincia'],
            'years_since_grand_opening_x':   ['ID del activo', '# de Tienda', 'Estado/Provincia'],
            'years_since_last_touch_x':      ['ID del activo', '# de Tienda', 'Estado/Provincia']
        }

        indicators_dataframes = []
        comments_indicators = []

        common = df[['Store_Asset_ID_x']].drop_duplicates().reset_index(drop=True).copy()
        indicators_dataframes.append(
            df[['Store_Asset_ID_x', group, '# de Tienda', 'Estado/Provincia']]
            .drop_duplicates(subset='Store_Asset_ID_x')
            .reset_index(drop=True)
            .copy()
        )

        for indicator, inv_proportional in indicators.items():
            if indicator != 'Fails_Frequency_Index':
                print('entra')
                print(indicator)
                strategy = 'standard'
                pre_df_indicator = self.get_filtered_data(df, indicators_and_fields_implied_relations[indicator])
                pre_df_indicator = self.map_indicator_to_health_using_logistic_function(
                    pre_df_indicator[['Store_Asset_ID_x', indicator + '_' + group]],
                    group, indicator, strategy, inv_proportional
                )
                print(pre_df_indicator.columns)
                pre_df_indicator = pre_df_indicator.drop_duplicates(subset='Store_Asset_ID_x').reset_index(drop=True)
                df_indicator = common.merge(pre_df_indicator, how = 'left', on='Store_Asset_ID_x')
            else:
                pre_df_indicator = df[['Store_Asset_ID_x', 'Fails_Frequency_Index_ID del activo',
                    'Health_Indicator_Based_On_Logistic_Of_Fails_Frequency_Index_ID del activo']].copy()
                #df_indicator = df[['Store_Asset_ID_x', group, 'Fails_Frequency_Index_ID del activo',
                #    'Health_Indicator_Based_On_Logistic_Of_Fails_Frequency_Index_ID del activo']]
                pre_df_indicator = pre_df_indicator[pre_df_indicator['Fails_Frequency_Index_ID del activo'].isna()]
                #pre_df_indicator = pre_df_indicator.drop_duplicates(subset=group)
                pre_df_indicator = pre_df_indicator.drop_duplicates(subset=['Store_Asset_ID_x'])
                pre_df_indicator = pre_df_indicator.reset_index(drop=True)
                #df_indicator = common.merge(pre_df_indicator, how='left', on=group)
                df_indicator = common.merge(pre_df_indicator, how='left', on='Store_Asset_ID_x')
                print('no entra')
                print(indicator)

            df_indicator = self.preprocess_indicator(df_indicator, indicator, group)
            #df_indicator.to_excel('df_indicator' + indicator + '.xlsx')
            indicators_dataframes.append(df_indicator)
            comments_indicators.append('Comments_' + indicator + '_' + group)
        #result = reduce(lambda left, right: pd.merge(left, right, on='ID del activo', how='inner'), indicators_dataframes)
        result = reduce(lambda left, right: pd.merge(left, right, on='Store_Asset_ID_x', how='inner'), indicators_dataframes)
        result['indicators_comments'] = result[comments_indicators] \
            .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        result['indicators_comments'] = result['indicators_comments'].str.replace(r'.{2,}', ' ', regex=True)
        result['indicators_comments'] = result['indicators_comments'].str.replace(r'^\s+', '', regex=True)
        result['indicators_comments'] = result['indicators_comments'].str.replace(r'\s+$', '', regex=True)
        result['indicators_comments'] = result['indicators_comments'].replace('', 'N/A')
        result = result.drop(columns=comments_indicators)
        self.logger.info('Success Execution of map_indicator_using_min_max_and_optimal_value_approach in health_indicators.py')
        return result

    def build_metric_percentil(self, pre_df, groups, indicators, percentiles):
        self.logger.info('Beginning Execution of build_metric_percentil in health_indicators.py')
        df = pre_df.copy()
        indicatorskeys = list(indicators.keys())
        if len(indicators.keys()) == 2:
            for group, percentile in product(groups, percentiles):
                #str_column_indicator1 = 'Health_Indicator_Based_On_' + indicatorskeys[0] + '_' + group + '_' + str(percentile)
                #str_column_indicator2 = 'Health_Indicator_Based_On_' + indicatorskeys[1] + '_' + group + '_' + str(percentile)
                str_column_indicator1 = (
                    'Health_Indicator_Based_On_Logistic_Of_'
                    + indicatorskeys[0] + '_' + group + '_' + str(percentile)
                )
                str_column_indicator2 = (
                    'Health_Indicator_Based_On_Logistic_Of_'
                    + indicatorskeys[1] + '_' + group + '_' + str(percentile)
                )
                str_column_metric = (
                    'Health_Metric_Based_On_Logistic_Of_'
                    + indicatorskeys[0]
                    + '_And_'
                    + indicatorskeys[1]
                    + '_'
                    + group
                    + '_'
                    + str(percentile)
                )
                weight_indicator1 = indicators[indicatorskeys[0]]
                weight_indicator2 = indicators[indicatorskeys[1]]
                df[str_column_metric] = (
                    df[str_column_indicator1] * weight_indicator1
                    + df[str_column_indicator2] * weight_indicator2
                )
                df = self.get_health_by_store_by_percentile(df, group, indicatorskeys, percentile)
        self.logger.info('Success Execution of build_metric_percentil in health_indicators.py')
        return df

    def build_metric(self, pre_df, groups, indicators):
        self.logger.info('Beginning Execution of build_metric in health_indicators.py')
        df = pre_df.copy()
        indicatorskeys = list(indicators.keys())
        indicator_count = len(indicatorskeys)
        weighted_asset_columns = []
        weighted_store_columns = []
        weighted_state_columns = []

        for i in range(indicator_count):
            str_asset_column_indicator = (
                'Health_Indicator_Based_On_Logistic_Of_{0}_{1}'
                .format(indicatorskeys[i], groups[0])
            )
            str_store_column_indicator = (
                'Health_Store_Based_On_{0}'
                .format(indicatorskeys[i])
            )
            #str_state_column_indicator = 'Health_State_Based_On_{0}'.format(indicatorskeys[i])
            weight_indicator = indicators[indicatorskeys[i]]

            df = self.get_partial_health_by_store(df, str_asset_column_indicator, indicatorskeys[i])
            df = self.get_partial_health_by_group(
                df, str_asset_column_indicator, indicatorskeys[i], 'Estado/Provincia'
            )

            df['Weighted_' + str_asset_column_indicator] = (
                df[str_asset_column_indicator] * weight_indicator
            )
            df['Weighted_Store_' + str_store_column_indicator] = (
                df[str_store_column_indicator] * weight_indicator
            )
            #df['Weighted_State_' + str_state_column_indicator] = df[str_state_column_indicator] * weight_indicator

            weighted_asset_columns.append('Weighted_' + str_asset_column_indicator)
            weighted_store_columns.append('Weighted_Store_' + str_store_column_indicator)
            #weighted_state_columns.append('Weighted_State_' + str_state_column_indicator)

        str_asset_column_metric = (
            'Health_Metric_Based_On_Logistic_Of_{0}_{1}'
            .format('_And_'.join(indicatorskeys), groups[0])
        )
        str_store_column_metric = (
            'Health_Store_Metric_Based_On_{0}'
            .format('_And_'.join(indicatorskeys))
        )
        #str_state_column_metric = (
        #    'Health_State_Metric_Based_On_{0}_{1}'
        #    .format('_And_'.join(indicatorskeys), groups[0])
        #)

        df[str_asset_column_metric] = df[weighted_asset_columns].sum(axis=1)
        df[str_store_column_metric] = df[weighted_store_columns].sum(axis=1)
        df = self.get_general_services_health_by_group(df, groups[0], '# de Tienda', indicatorskeys)
        df = self.get_general_services_health_by_group(df, groups[0], 'Estado/Provincia', indicatorskeys, True)
        self.logger.info('Success Execution of build_metric in health_indicators.py')
        return df

    def get_partial_health_by_group(self, pre_df, health_indicator, indicator, group, target_name=None):
        self.logger.info('Beginning Execution of get_partial_health_by_group in health_indicators.py')
        df = pre_df.copy()
        if group == 'Estado/Provincia':
            groupstr='State'
        df['peso_inverso_'+health_indicator] = 1/df[health_indicator]
        df['peso_inverso_sum_'+health_indicator] = df.groupby(group)['peso_inverso_'+health_indicator].transform(sum)
        df['peso_normalizado_'+health_indicator] = df['peso_inverso_'+health_indicator] / df['peso_inverso_sum_'+health_indicator]
        df['premedia_ponderada_'+health_indicator] = df[health_indicator] * df['peso_normalizado_'+health_indicator]
        if target_name == None:
            df['Health_'+groupstr+'_Based_On_'+indicator] = df.groupby(group)['premedia_ponderada_'+health_indicator].transform(sum)
            df['Health_'+groupstr+'_Based_On_'+indicator] = df['Health_'+groupstr+'_Based_On_'+indicator].round(4)
        else:
            df[target_name] = df.groupby(group)['premedia_ponderada_'+health_indicator].transform(sum)
            df[target_name] = df[target_name].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_'+health_indicator,
            'peso_inverso_sum_'+health_indicator,
            'peso_normalizado_'+health_indicator,
            'premedia_ponderada_'+health_indicator
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        self.logger.info('Success Execution of get_partial_health_by_group in health_indicators.py')
        return df

    def get_partial_health_by_store(self, pre_df, health_indicator, indicator):
        self.logger.info('Beginning Execution of get_partial_health_by_store in health_indicators.py')
        df = pre_df.copy()
        df['peso_inverso_'+health_indicator] = 1/df[health_indicator]
        df['peso_inverso_sum_'+health_indicator] = df.groupby('# de Tienda')['peso_inverso_'+health_indicator].transform(sum)
        df['peso_normalizado_'+health_indicator] = df['peso_inverso_'+health_indicator] / df['peso_inverso_sum_'+health_indicator]
        df['premedia_ponderada_'+health_indicator] = df[health_indicator] * df['peso_normalizado_'+health_indicator]
        df['Health_Store_Based_On_'+indicator] = df.groupby('# de Tienda')['premedia_ponderada_'+health_indicator].transform(sum)
        df['Health_Store_Based_On_'+indicator] = df['Health_Store_Based_On_'+indicator].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_'+health_indicator,
            'peso_inverso_sum_'+health_indicator,
            'peso_normalizado_'+health_indicator,
            'premedia_ponderada_'+health_indicator
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        self.logger.info('Success Execution of get_partial_health_by_store in health_indicators.py')
        return df

    def get_general_health_by_group(self, pre_df, group, granularity, indicatorskeys, map_indicator=False):
        df = pre_df.copy()
        if granularity == 'Estado/Provincia':
            granularityStr = 'State'
        elif granularity == '# de Tienda':
            granularityStr = 'Store'
        suffix = '_And_'.join(indicatorskeys) + '_' + group if indicatorskeys else ''
        # suffix = '_And_'.join(indicatorskeys) if indicatorskeys else ''
        print('suffix:', 'Health_Metric_Based_On_Logistic_Of_' + suffix)
        df['peso_inverso_' + suffix] = 1 / df['Health_Metric_Based_On_Logistic_Of_' + suffix]
        df['peso_inverso_sum_' + suffix] = df.groupby(granularity)['peso_inverso_' + suffix].transform(sum)
        df['peso_normalizado_' + suffix] = df['peso_inverso_' + suffix] / df['peso_inverso_sum_' + suffix]
        df['premedia_ponderada_' + suffix] = df['Health_Metric_Based_On_Logistic_Of_' + suffix] * df['peso_normalizado_' + suffix]
        df['Health_' + granularityStr + '_Based_On_' + suffix] = df.groupby(granularity)['premedia_ponderada_' + suffix].transform(sum)
        df['Health_' + granularityStr + '_Based_On_' + suffix] = df['Health_' + granularityStr + '_Based_On_' + suffix].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_' + suffix,
            'peso_inverso_sum_' + suffix,
            'peso_normalizado_' + suffix,
            'premedia_ponderada_' + suffix
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        if map_indicator:
            df = self.map_indicator_to_health_using_logistic_function(
                df, group,
                'Health_' + granularityStr + '_Based_On_' + suffix,
                'standard', False
            )
        self.logger.info('Success Execution of get_partial_health_by_store in health_indicators.py')
        return df

    def get_general_services_health_by_group(self, pre_df, group, granularity, health_asset_column, map_indicator=False):
        self.logger.info('Beginning Execution of get_general_services_health_by_group in health_indicators.py')
        df = pre_df.copy()
        if granularity == 'Estado/Provincia':
            granularityStr = 'State'
        elif granularity == '# de Tienda':
            granularityStr = 'Store'
        suffix = granularityStr
        # print('suffix:', 'Health_Metric_Based_On_Logistic_Of_' + suffix)
        df['peso_inverso_' + suffix] = 1 / df[health_asset_column]
        df['peso_inverso_sum_' + suffix] = df.groupby(granularity)['peso_inverso_' + suffix].transform(sum)
        df['peso_normalizado_' + suffix] = df['peso_inverso_' + suffix] / df['peso_inverso_sum_' + suffix]
        df['premedia_ponderada_' + suffix] = df[health_asset_column] * df['peso_normalizado_' + suffix]
        df['Health_Services_' + granularityStr] = df.groupby(granularity)['premedia_ponderada_' + suffix].transform(sum)
        df['Health_Services_' + granularityStr] = df['Health_Services_' + granularityStr].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_' + suffix,
            'peso_inverso_sum_' + suffix,
            'peso_normalizado_' + suffix,
            'premedia_ponderada_' + suffix
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        if map_indicator:
            df = self.map_indicator_to_health_using_logistic_function(
                df, group,
                'Health_Services_' + granularityStr,
                'standard', False
            )
        self.logger.info('Success Execution of get_general_services_health_by_group in health_indicators.py')
        return df

    def get_general_health_by_group(self, pre_df, group, granularity, indicatorskeys, map_indicator=False):
        df = pre_df.copy()
        if granularity == 'Estado/Provincia':
            granularityStr = 'State'
        elif granularity == '# de Tienda':
            granularityStr = 'Store'
        suffix = '_And_'.join(indicatorskeys) + '_' + group if indicatorskeys else ''
        # suffix = '_And_'.join(indicatorskeys) if indicatorskeys else ''
        print('suffix:', 'Health_Metric_Based_On_Logistic_Of_' + suffix)
        df['peso_inverso_' + suffix] = 1 / df['Health_Metric_Based_On_Logistic_Of_' + suffix]
        df['peso_inverso_sum_' + suffix] = df.groupby(granularity)['peso_inverso_' + suffix].transform(sum)
        df['peso_normalizado_' + suffix] = df['peso_inverso_' + suffix] / df['peso_inverso_sum_' + suffix]
        df['premedia_ponderada_' + suffix] = df['Health_Metric_Based_On_Logistic_Of_' + suffix] * df['peso_normalizado_' + suffix]
        df['Health_' + granularityStr + '_Based_On_' + suffix] = df.groupby(granularity)['premedia_ponderada_' + suffix].transform(sum)
        df['Health_' + granularityStr + '_Based_On_' + suffix] = df['Health_' + granularityStr + '_Based_On_' + suffix].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_' + suffix,
            'peso_inverso_sum_' + suffix,
            'peso_normalizado_' + suffix,
            'premedia_ponderada_' + suffix
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        if map_indicator:
            df = self.map_indicator_to_health_using_logistic_function(
                df, group,
                'Health_' + granularityStr + '_Based_On_' + suffix,
                'standard', False
            )
        self.logger.info('Success Execution of get_partial_health_by_store in health_indicators.py')
        return df

    def get_general_services_health_by_group(self, pre_df, group, granularity, health_asset_column, map_indicator=False):
        self.logger.info('Beginning Execution of get_general_services_health_by_group in health_indicators.py')
        df = pre_df.copy()
        if granularity == 'Estado/Provincia':
            granularityStr = 'State'
        elif granularity == '# de Tienda':
            granularityStr = 'Store'
        suffix = granularityStr
        # print('suffix:', 'Health_Metric_Based_On_Logistic_Of_' + suffix)
        df['peso_inverso_' + suffix] = 1 / df[health_asset_column]
        df['peso_inverso_sum_' + suffix] = df.groupby(granularity)['peso_inverso_' + suffix].transform(sum)
        df['peso_normalizado_' + suffix] = df['peso_inverso_' + suffix] / df['peso_inverso_sum_' + suffix]
        df['premedia_ponderada_' + suffix] = df[health_asset_column] * df['peso_normalizado_' + suffix]
        df['Health_Services_' + granularityStr] = df.groupby(granularity)['premedia_ponderada_' + suffix].transform(sum)
        df['Health_Services_' + granularityStr] = df['Health_Services_' + granularityStr].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_' + suffix,
            'peso_inverso_sum_' + suffix,
            'peso_normalizado_' + suffix,
            'premedia_ponderada_' + suffix
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        if map_indicator:
            df = self.map_indicator_to_health_using_logistic_function(
                df, group,
                'Health_Services_' + granularityStr,
                'standard', False
            )
        self.logger.info('Success Execution of get_general_services_health_by_group in health_indicators.py')
        return df

    def get_general_health_by_store(self, pre_df, group, indicatorskeys):
        self.logger.info('Beginning Execution of get_general_health_by_store in health_indicators.py')
        df = pre_df.copy()
        suffix = '_And_'.join(indicatorskeys) + '_' + group if indicatorskeys else ''
        print('suffix:', 'Health_Metric_Based_On_Logistic_Of_' + suffix)
        df['peso_inverso_' + suffix] = 1 / df['Health_Metric_Based_On_Logistic_Of_' + suffix]
        df['peso_inverso_sum_' + suffix] = df.groupby('# de Tienda')['peso_inverso_' + suffix].transform(sum)
        df['peso_normalizado_' + suffix] = df['peso_inverso_' + suffix] / df['peso_inverso_sum_' + suffix]
        df['premedia_ponderada_' + suffix] = df['Health_Metric_Based_On_Logistic_Of_' + suffix] * df['peso_normalizado_' + suffix]
        df['Health_Store_Based_On_' + suffix] = df.groupby('# de Tienda')['premedia_ponderada_' + suffix].transform(sum)
        df['Health_Store_Based_On_' + suffix] = df['Health_Store_Based_On_' + suffix].round(4)
        not_verbose_filtered_columns = [
            'peso_inverso_' + suffix,
            'peso_inverso_sum_' + suffix,
            'peso_normalizado_' + suffix,
            'premedia_ponderada_' + suffix
        ]
        df = df[[col for col in df.columns if col not in not_verbose_filtered_columns]]
        self.logger.info('Success Execution of get_general_health_by_store in health_indicators.py')
        return df

    def get_tmef_analysis(self, pre_df, target_date_column, group, strategy):
        self.logger.info('Beginning Execution of get_tmef_analysis in health_indicators.py')
        groupStr = '_'.join([x.replace('#', '').replace(' ', '') for x in group])
        df = pre_df.copy()
        if strategy == 'with_today':
            df['Termino Real'] = df['Termino Real'].astype('datetime64[ns]')
            df['Termino Real'] = pd.to_datetime(df['Termino Real']).dt.date
            fecha_hoy = date.today()
            df_hoy = pd.DataFrame({group: df[group].unique(), target_date_column: fecha_hoy})
            temp = pd.concat([df, df_hoy], ignore_index=True)
            temp['Termino Real'] = temp['Termino Real'].astype(str)
            temp['Termino Real'] = temp['Termino Real'].astype('datetime64[ns]')
            temp['diff_' + target_date_column + '_' + group] = temp.groupby(group)[target_date_column].diff(periods=-1).dt.days
            temp['diff_' + target_date_column + '_' + group] = temp['diff_' + target_date_column + '_' + group].abs()
            temp['diff_' + target_date_column + '_' + group] = temp['diff_' + target_date_column + '_' + group] + 1  # Smooth of dates with zero days of difference
            temp = temp[temp[target_date_column] != fecha_hoy]
            temp['TMEF_' + target_date_column + '_' + group] = temp.groupby(group)['diff_' + target_date_column + '_' + group].transform('mean')
            # Debido a que se agrego otro renglón para cada asset se elimina donde no tenga información el identificador de asset
            temp = temp[~temp['Store_Asset_ID'].isna()]
            temp = temp.drop_duplicates(subset=['ID del activo'])
            temp = temp.drop(columns=['Termino Real', 'diff_' + target_date_column + '_' + group])
            self.logger.info('Success Execution of get_tmef_analysis in health_indicators.py')
            return temp
        elif strategy == 'without_today':
            df['Termino Real'] = df['Termino Real'].astype(str)
            df['Termino Real'] = df['Termino Real'].astype('datetime64[ns]')
            df = df.sort_values(by='Termino Real')
            df['diff_' + target_date_column + '_' + group] = df.groupby(group)[target_date_column].diff().dt.days
            # filter diff with nan values
            df = df[~df['diff_' + target_date_column + '_' + group].isna()]
            df['diff_' + target_date_column + '_' + group] = df['diff_' + target_date_column + '_' + group] + 1  # Smooth of dates with zero days of difference
            df['TMEF_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('mean')
            self.logger.info('Success Execution of get_tmef_analysis in health_indicators.py')
            return df

    def get_mdc_analysis(self, pre_df, target_date_column, target_cost_column, group):
        self.logger.info('Beginning Execution of get_mdc_analysis in health_indicators.py')
        groupStr = '_'.join([x.replace('#','').replace(' ','') for x in group])
        df = pre_df.copy()
        df['MDC_Parcial_' + group] = df[target_cost_column].divide(df['diff_' + target_date_column + '_' + group])
        print('method creating column: ' + target_cost_column + '_' + group +
              ' equals to sum of ' + target_cost_column + ' and grouped by ' + group)
        df[target_cost_column + '_' + group] = df.groupby(group)[target_cost_column].transform('sum')
        df['sum_diff_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('sum')
        df['MDC_Total_' + group] = df[target_cost_column + '_' + group].divide(df['sum_diff_' + target_date_column + '_' + group])
        # df['MDC_Total_' + group] = df['MDC_Total_' + group].divide(df['count_ID_de_Tarea_x_ID_del_activo'])
        df['MDC_Parcial_' + group] = df[target_cost_column].divide(df['diff_' + target_date_column + '_' + group])
        df['std_TMEF_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('std')
        df['TMEF_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('mean')
        self.logger.info('Success Execution of get_mdc_analysis in health_indicators.py')
        return df

    def get_tmef_analysis_old(self, pre_df, target_date_column, group):
        groupStr = '_'.join([x.replace('#','').replace(' ','') for x in group])
        df = pre_df.copy()
        # df['sum_diff_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('sum')
        # df['std_TMEF_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('std')
        # df['TMEF_' + target_date_column + '_' + group] = df.groupby(group)['diff_' + target_date_column + '_' + group].transform('mean')
        return df

    def get_filtered_data(self, pre_df, tmef_columns_implied):
        df = pre_df.copy()
        df['field_implied_in_data_comments'] = df['field_implied_in_data_comments'].fillna('N/A')
        df = df[
            df['field_implied_in_data_comments']
              .apply(lambda x: bool(set(x.split(',')) & set(tmef_columns_implied)))
        ].reset_index(drop=True)
        return df

    def get_raw_indicators(self, pre_df, group, strategy_deltas, target_date_column, last_months):
        self.logger.info('Beginning Execution of get_mdc_analysis in health_indicators.py')
        df = pre_df.copy()
        df_before_tmef = pre_df.copy()
        tmef_columns_implied = ['ID del activo', '# de Tienda', 'Estado/Provincia', target_date_column]
        df_tmef_filtered = self.get_filtered_data(df_before_tmef, tmef_columns_implied)
        df_tmef = self.get_tmef_analysis(
            df_tmef_filtered[['Store_Asset_ID', 'ID del activo', target_date_column]],
            target_date_column, group, strategy_deltas
        )
        df_tmef_result = df.merge(df_tmef, how='left', on='ID del activo')
        df_before_fail_index = pre_df.copy()
        index_fail_column_implied = ['ID del activo', '# de Tienda', 'Estado/Provincia', target_date_column]
        df_index_filtered = self.get_filtered_data(df_before_fail_index, index_fail_column_implied)
        df_fail_index = self.get_ots_analysis_cumulated_approach(df_index_filtered, last_months)
        df_raw_indicators = df_tmef_result.merge(df_fail_index, how='left', on='ID del activo')
        self.logger.info('Success Execution of get_mdc_analysis in health_indicators.py')
        return df_raw_indicators

    def get_mdc_indicator(self, pre_df: pd.DataFrame, group: str, strategy_deltas: str,
                          target_date_column: str, target_cost_column: str) -> pd.DataFrame:
        self.logger.info('Beginning Execution of get_mdc_analysis in health_indicators.py')
        # No se considera Fecha de cierre de la OT porque tiene valores nulos
        df = pre_df.copy()
        df = self.get_deltas(df, target_date_column, group, strategy_deltas)
        df = self.get_mdc_analysis(df, target_date_column, target_cost_column, group)
        self.logger.info('Success Execution of get_mdc_analysis in health_indicators.py')
        return df

    def get_binning_analysis(self, pre_df, columnTarget):
        self.logger.info('Beginning Execution of get_binning_analysis in health_indicators.py')
        df = pre_df.copy()
        quantiles, bins = pd.qcut(df[columnTarget], q=5, retbins=True)
        labels = [
            f'0.2_MuyBajo_[{bins[0]:.2f}-{bins[1]:.2f}]',
            f'0.4_Bajo_[{bins[1]:.2f}-{bins[2]:.2f}]',
            f'0.6_Medio_[{bins[2]:.2f}-{bins[3]:.2f}]',
            f'0.8_Alto_[{bins[3]:.2f}-{bins[4]:.2f}]',
            f'1_MuyAlto_[{bins[4]:.2f}-{bins[5]:.2f}]'
        ]
        df['binning_' + columnTarget] = pd.qcut(df[columnTarget], q=5, labels=labels)
        # print('binned ' + columnTarget)
        self.logger.info('Success Execution of get_binning_analysis in health_indicators.py')
        return df
    
    def get_decil_groups(self, pre_df, columnTarget):
        self.logger.info('Beginning Execution of get_decil_groups in health_indicators.py')
        df = pre_df.copy()
        deciles = pd.qcut(df[columnTarget], q=10, duplicates='drop')
        labels = [
            f'{int(interval.left)}-{int(interval.right)}'
            for interval in deciles.dtype.categories
        ]
        df['decil_groups_' + columnTarget] = pd.qcut(
            df[columnTarget],
            q=10,
            labels=labels
        )
        self.logger.info('Success Execution of get_decil_groups in health_indicators.py')
        return df

    def get_KMeans_analysis(self, df, n, columnTarget):
        self.logger.info('Beginning Execution of get_KMeans_analysis in health_indicators.py')
        kmeans = KMeans(n_clusters=n, random_state=0)
        X = df[columnTarget].values.reshape(-1,1)
        df['cluster_' + columnTarget] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_.flatten()
        sorted_clusters = np.argsort(centroids)
        rango_clusters = {}
        for cluster in sorted_clusters:
            cluster_values = df[df['cluster_' + columnTarget] == cluster][columnTarget]
            min_val = cluster_values.min()
            max_val = cluster_values.max()
            rango_clusters[cluster] = f'{min_val:.2f}-{max_val:.2f}'
        if n == 7:
            etiquetas_cluster = {
                sorted_clusters[0]: f'MuyBajo_[{rango_clusters[sorted_clusters[0]]}]',
                sorted_clusters[1]: f'MedianamenteBajo_[{rango_clusters[sorted_clusters[1]]}]',
                sorted_clusters[2]: f'Bajo_[{rango_clusters[sorted_clusters[2]]}]',
                sorted_clusters[3]: f'Medio_[{rango_clusters[sorted_clusters[3]]}]',
                sorted_clusters[4]: f'Alto_[{rango_clusters[sorted_clusters[4]]}]',
                sorted_clusters[5]: f'MedianamenteAlto_[{rango_clusters[sorted_clusters[5]]}]',
                sorted_clusters[6]: f'MuyAlto_[{rango_clusters[sorted_clusters[6]]}]'
            }
        elif n == 5:
            etiquetas_cluster = {
                sorted_clusters[0]: f'MuyBajo_[{rango_clusters[sorted_clusters[0]]}]',
                sorted_clusters[1]: f'Bajo_[{rango_clusters[sorted_clusters[1]]}]',
                sorted_clusters[2]: f'Medio_[{rango_clusters[sorted_clusters[2]]}]',
                sorted_clusters[3]: f'Alto_[{rango_clusters[sorted_clusters[3]]}]',
                sorted_clusters[4]: f'MuyAlto_[{rango_clusters[sorted_clusters[4]]}]'
            }
        elif n == 4:
            etiquetas_cluster = {
                sorted_clusters[0]: f'MuyBajo_[{rango_clusters[sorted_clusters[0]]}]',
                sorted_clusters[1]: f'Bajo_[{rango_clusters[sorted_clusters[1]]}]',
                sorted_clusters[2]: f'Medio_[{rango_clusters[sorted_clusters[2]]}]',
                sorted_clusters[3]: f'Alto_[{rango_clusters[sorted_clusters[3]]}]'
            }
        elif n == 3:
            etiquetas_cluster = {
                sorted_clusters[0]: f'Bajo_[{rango_clusters[sorted_clusters[0]]}]',
                sorted_clusters[1]: f'Medio_[{rango_clusters[sorted_clusters[1]]}]',
                sorted_clusters[2]: f'Alto_[{rango_clusters[sorted_clusters[2]]}]'
            }
        df['binnedByClustering_'+columnTarget] = df['cluster_'+columnTarget].map(etiquetas_cluster)
        self.logger.info('Success Execution of get_KMeans_analysis in health_indicators.py')
        return df

