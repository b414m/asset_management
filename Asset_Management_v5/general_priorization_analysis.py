import plotly_express as px
import pandas as pd
import numpy as np
from datetime import datetime
from data_preprocessing.preprocessing import Preprocessing
from data_ingestion.data_loader import DataLoader
from data_transformation.health_indicators import HealthIndicators
from data_ingestion.sql_data_connection import SQLDataConnection
from data_transformation.priorization import Priorization
from data_quality.data_quality import DataQuality

class GeneralPriorizationAnalysis:
    def __init__(self, main_configuration_execution, logger):
        global df, dl, prepreoc, hi, prio, dq, today, valid_store_formats, priority_systems, store_number, request_class, asset_name, asset_id
        self.main_configuration_execution = main_configuration_execution
        priority_systems = [system for system, _ in self.main_configuration_execution['systems'][0].items()]
        valid_store_formats = self.main_configuration_execution['valid_store_formats']
        store_number = self.main_configuration_execution['fields_input_assets_requirements']['store_number']
        request_class = self.main_configuration_execution['fields_input_assets_requirements']['request_class']
        asset_name = self.main_configuration_execution['fields_input_assets_requirements']['asset_name']
        asset_id = self.main_configuration_execution['fields_input_assets_requirements']['asset_id']
        self.logger = logger
        connection_prod = SQLDataConnection(self.main_configuration_execution['input_assets_conn_params'], logger)
        dl = DataLoader(connection_prod, logger)
        prepreoc = Preprocessing(self.main_configuration_execution, logger)
        hi = HealthIndicators(logger)
        prio = Priorization(logger)
        dq = DataQuality(logger)
        today = datetime.now()

    def get_assets(self):
        df_assets = dl.load_data(None, 'sql-tririga')
        df_assets = df_assets[df_assets[store_number].isna()]
        return df_assets

    def only_get_age(self, df_assets, last_touch_date, service_name):
        df_last_touch = pd.read_excel('data_sources/OOC_Update.xlsx')
        df_last_touch = df_last_touch.rename(columns={'Lst Touch': 'Lst Touch OOC'})
        df_last_touch_updated = pd.read_excel('data_sources/mm_executed.xlsx')
        df_last_touch_updated = df_last_touch_updated.rename(
            columns={'Lst Touch': 'Lst Touch updated', 'Refri': 'Refrigeracion'}
        )
        df_last_touch_updated['Aire'] = df_last_touch_updated['Aire'].fillna('')
        df_last_touch_updated['Aire'] = df_last_touch_updated['Aire'].replace('No tiene Aire', '')
        df_last_touch_updated['Refrigeracion'] = df_last_touch_updated['Refrigeracion'].fillna('')
        df_last_touch_updated['Flag_Lst Touch Refrigeracion'] = np.where(
            df_last_touch_updated['Refrigeracion'] == '', False, True
        )
        df_last_touch_updated['Flag_Lst Touch Aire'] = np.where(
            df_last_touch_updated['Aire'] == '', False, True
        )

        stores = self.get_store_data(valid_store_formats)
        stores = stores.rename(columns={'GRAND_OPENING': 'GRAND_OPENING_x'})
        stores['GRAND_OPENING_x'] = pd.to_datetime(
            stores['GRAND_OPENING_x']
        ).dt.tz_localize(None)

        pre_aged_stores = stores[['STORE_NUMBER', 'GRAND_OPENING_x']].merge(
            df_last_touch[['Determinante', 'Lst Touch OOC']],
            how='left',
            left_on='STORE_NUMBER',
            right_on='Determinante'
        )
        pre_aged_stores['Lst Touch OOC'] = pre_aged_stores['Lst Touch OOC'].fillna('')
        pre_aged_stores['Flag_Lst Touch OOC'] = np.where(
            pre_aged_stores['Lst Touch OOC'] == '', False, True
        )
        pre_aged_stores['Lst Touch OOC'] = np.where(
            pre_aged_stores['Lst Touch OOC'] == '',
            pre_aged_stores['GRAND_OPENING_x'].dt.year,
            pre_aged_stores['Lst Touch OOC']
        )
        pre_aged_stores = pre_aged_stores.drop(columns=['Determinante'])

        aged_stores = pre_aged_stores.merge(
            df_last_touch_updated[[
                'Determinante', 'Lst Touch updated',
                'Flag_Lst Touch Refrigeracion', 'Refrigeracion',
                'Flag_Lst Touch Aire', 'Aire'
            ]],
            how='left',
            left_on='STORE_NUMBER',
            right_on='Determinante'
        )
        aged_stores['Aire'] = np.where(
            aged_stores['Flag_Lst Touch Aire'] == False,
            aged_stores['Aire'],
            aged_stores['Lst Touch OOC']
        )
        aged_stores['Refrigeracion'] = np.where(
            aged_stores['Flag_Lst Touch Refrigeracion'] == False,
            aged_stores['Refrigeracion'],
            aged_stores['Lst Touch OOC']
        )
        aged_stores['Aire'] = aged_stores['Aire'].fillna('')
        aged_stores['Refrigeracion'] = aged_stores['Refrigeracion'].fillna('')
        aged_stores['rest'] = np.where(
            (aged_stores['Refrigeracion'] != '') & (aged_stores['Aire'] != ''),
            aged_stores['Lst Touch OOC'],
            None
        )
        aged_stores['all'] = np.where(
            (aged_stores['Refrigeracion'] != '') & (aged_stores['Aire'] != ''),
            aged_stores['Lst Touch OOC'],
            None
        )
        # aged_stores.to_excel('Previous.xlsx')
        result = aged_stores.melt(
            id_vars=['STORE_NUMBER'],
            value_vars=['Refrigeracion', 'Aire', 'rest', 'all'],
            var_name='Service',
            value_name='Lst Touch'
        )
        result = result[result['Lst Touch'] != '']
        result = result[result['Lst Touch'].isna()]
        result['Lst Touch'] = result['Lst Touch'].astype(int).astype(str)

        pre_final = result.merge(
            stores[['STORE_NUMBER', 'GRAND_OPENING_x', 'BANNER', 'NOMBRE_ENT_DEP']],
            how='left',
            on='STORE_NUMBER'
        )
        pre_final['Lst Touch Normalized'] = pd.to_datetime(
            pre_final['Lst Touch'],
            format='%Y-%m-%d'
        )
        pre_final['today'] = today
        pre_final['years_since_last_touch_x_ID del activo'] = (
            pre_final['today'] - pre_final['Lst Touch Normalized']
        ).dt.days / 365.25
        pre_final['years_since_grand_opening_x_ID del activo'] = (
            pre_final['today'] - pre_final['GRAND_OPENING_x']
        ).dt.days / 365.25
        pre_final['years_since_grand_opening_x_ID del activo'] = (
        pre_final['today'] - pre_final['GRAND_OPENING_x']
        ).dt.days / 365.25
        pre_final['Service'] = pre_final['Service'].str.replace('all', service_name)
        pre_final = pre_final[pre_final['Service'] == service_name]
        pre_final['STORE_NUMBER'] = pre_final['STORE_NUMBER'].astype(int)
        final = df_assets.merge(
            pre_final,
            how='left',
            right_on='STORE_NUMBER',
            left_on='# de Tienda'
        )
        final['years_since_grand_opening_x_ID del activo'] = final[
            'years_since_grand_opening_x_ID del activo'
        ].fillna(0)
        final['years_since_last_touch_x_ID del activo'] = final[
            'years_since_last_touch_x_ID del activo'
        ].fillna(0)
        final['# de Tienda'] = np.where(
            final['# de Tienda'].isna(),
            final['STORE_NUMBER'],
            final['# de Tienda']
        )
        final['Formato de negocio'] = np.where(
            final['Formato de negocio'].isna(),
            final['BANNER'],
            final['Formato de negocio']
        )
        final['Estado/Provincia'] = np.where(
            final['Estado/Provincia'].isna(),
            final['NOMBRE_ENT_DEP'],
            final['Estado/Provincia']
        )
        final['Estado/Provincia'] = final['Estado/Provincia'].replace(
            'Distrito Federal',
            'Ciudad de México'
        )
        final = prepreoc.preprocessing_states(final)
        final = final[final['Formato de negocio'].isin(valid_store_formats)]
        final.to_excel('1_Debugging_after_filter_valid_store_formats.xlsx')
        final['Created Date'] = pd.to_datetime(final['Created Date'])
        final['ot_after_last_touch_flag'] = np.where(
            final['Created Date'] > final['Lst Touch Normalized'],
            True,
            False
        )
        final = final[final['ot_after_last_touch_flag'] == True]
        final['Store_Asset_ID'] = (
            final['# de Tienda'].astype(int).astype(str)
            + '_' 
            + final['ID del activo'].astype(str).fillna('')
        )
        final.to_excel('2_Debugging_after_filter_ots_last_touch_flag.xlsx')
        #final = final.drop(columns=['Service'])
        #stores.to_excel('Stores.xlsx')
        return final, stores


    def get_services_and_subsystems_by_TRIRIGA(self, pre_df):
        self.logger.info(
            'Beginning execution of get_services_and_subsystems_by_TRIRIGA method'
        )
        df = pre_df.copy()
        df['Request Class Preproc'] = df[request_class]
        df = prepreoc.text_normalized(df, 'Request Class Preproc')
        request_class_dict = df.drop_duplicates(
            'Request Class Preproc'
        ).set_index('Request Class Preproc')[request_class].to_dict()
        df['Pre-Service'] = np.where(
            df[request_class].isna(),
            df[asset_name],
            df[request_class]
        )
        df = prepreoc.filter_request_class(df)
        df = prepreoc.text_normalized(df, 'Pre-Service')
        df['Pre-Service'] = df['Pre-Service'].replace(request_class_dict)
        df = df.dropna(subset=['Pre-Service'])
        df['Service'] = df['Pre-Service'].apply(lambda x: x.split('-')[0])
        df['Service'] = df['Service'].str.replace(r' $', '', regex=True)
        df = df[df['Service'].isin(priority_systems)]
        df['Store_Asset_ID'] = (
            df[store_number].astype(int).astype(str)
            + '_' 
            + df[asset_id]
        )
        self.logger.info(
            'Success execution of get_services_and_subsystems_by_TRIRIGA method'
        )
        return df

    def run(self):
        self.logger.info('Begining execution of run method in general_priorization_analysis.py file')
        health_asset_column = 'General_Health_Asset_Indicator'
        health_store_column = 'General_Health_Store_Indicator'
        health_state_column = 'General_Health_State_Indicator'
        services_weights = {'Refrigeracion': 0.8, 'Aire': 0.2}
        df = self.get_assets()
        df_tririga = self.get_services_and_subsystems_by_TRIRIGA(df)
        result = df_tririga.groupby('Service') \
            .apply(lambda x: self.get_health_analysis(x, x.name))
        result.to_excel('DebuggingResults.xlsx')
        result = result.reset_index()
        result = dq.get_cases(result)
        #result = hm.get_general_services_health(result, health_asset_column)
        #result = hm.get_general_services_healthv2(result, health_store_column, services_weights, '# de Tienda')
        #result = hm.get_general_services_healthv2(result, health_state_column, services_weights, 'Estado/Provincia')
        self.logger.info('Success execution of run method in general_priorization_analysis.py file')
        return result

    def get_health_analysis(self, pre_df, service_name):
        self.logger.info('Begining execution of get_health_analysis method')
        health_asset_column = 'General_Health_Asset_Indicator'
        df = pre_df.copy()
        strategy_deltas = '_with_today'
        df_preproc = prepreoc.general_preprocessing(df)
        df, stores = self.only_get_age(df_preproc, '12-31', service_name)
        self.logger.info('success get_age')
        df = hi.get_indicators(df, strategy_deltas)
        self.logger.info('success get raw Indicators')
        df = hi.get_indicators_analysis(df)
        self.logger.info('success get_indicators_analysis')
        result = stores[['STORE_NUMBER']].merge(df, how='left', on='STORE_NUMBER')
        result[store_number] = np.where(
            result[store_number].isna(),
            result['STORE_NUMBER'],
            result[store_number]
        )
        result = result.drop(columns=['STORE_NUMBER'])
        result = prepreoc.rename_output_fields(result)
        result['Store_Asset_ID'] = (
            result[store_number].astype(int).astype(str)
            + '_' + result[asset_id]
        )
        final = result.merge(
            df_preproc[['Store_Asset_ID', 'Pre-Service']].drop_duplicates(),
            how='left',
            on='Store_Asset_ID'
        )
        final = prio.get_priorization_by_request_class_group(
            final, service_name, health_asset_column
        )
        self.logger.info('Success execution of get_health_analysis method')
        return final

    def get_store_data(self, valid_store_formats):
        df_stores = dl.load_data(None, 'databricks-lat')
        df_stores = prepreoc.preprocessing_lat_stores(df_stores)
        return df_stores

    def get_priorization(self, pre_df, service_name):
        health_asset_column = 'General_Health_Asset_Indicator'
        importance_df = pd.read_excel('data_sources/service_importance.xlsx')
        importance_dict = (
            importance_df[importance_df['Servicio'] == service_name]
            .set_index('Subsistema')
            .to_dict()['Importancia Definida por Negocio']
        )
        category_dict = (
            importance_df[
                ['Categoría Importancia Definida por Negocio', 'Importancia Definida por Negocio']
            ]
            .drop_duplicates()
            .set_index('Importancia Definida por Negocio')
            .to_dict()['Categoría Importancia Definida por Negocio']
        )
        df = pre_df.copy()
        final = prio.get_priorization_by_request_class_group(df, importance_dict, health_asset_column)
        final['Category_Group_Priorization_Asset_Index'] = final['Importance_Group'].replace(category_dict)
        # final['Category_Group_Priorization_Asset_Index'] = pd.cut(
        #     final['Group_Priorization_Asset_Index'],
        #     bins=5,
        #     labels=['Critico Alto', 'Critico Medio', 'Critico Bajo', 'Advertencia', 'Correcto']
        # )
        # final = hm.map_indicator_to_health_using_logistic_function(
        #     final, '', 'Group_Priorization_Asset_Index', 'standard', False
        # )
        return final

