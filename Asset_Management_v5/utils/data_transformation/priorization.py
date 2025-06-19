import pandas as pd
import numpy as np
from .health_indicators import HealthIndicators

class Priorization:
    def __init__(self, logger):
        self.logger = logger
        global hi
        hi = HealthIndicators(logger)

    def get_priorization_by_request_class_group(self, pre_df, service_name, health_asset_indicator):
        self.logger.info('Beginning Execution of get_priorization_by_request_class_group method in priorization.py')
        df = pre_df.copy()
        # health_asset_column = 'General_Health_Asset_Indicator'
        importance_df = pd.read_excel('data_sources/service_importance.xlsx')
        importance_dict = (
            importance_df[importance_df['Servicio'] == service_name]
              .set_index('Subsistema')
              .to_dict()['Importancia Definida por Negocio']
        )
        category_dict = (
            importance_df[
                ['Categoría Importancia Definida por Negocio','Importancia Definida por Negocio']
            ]
            .drop_duplicates()
            .set_index('Importancia Definida por Negocio')
            .to_dict()['Categoría Importancia Definida por Negocio']
        )
        columnTarget = 'Pre-Service'
        df['Importance_Group'] = df[columnTarget].replace(importance_dict)
        df['Category_Group_Priorization_Asset_Index'] = df['Importance_Group'].replace(category_dict)
        orden = df.sort_values(
            ['Importance_Group', health_asset_indicator],
            ascending=[True, True]
        ).index
        # df = df.sort_values(columnTarget)
        # df['Group_Priorization_Asset_Index'] = df.groupby(columnTarget).cumcount()
        # orden = df.sort_values(
        #     ['Group_Priorization_Asset_Index', health_asset_indicator],
        #     ascending=[True, True]
        # ).index
        # df.loc[orden, 'General_Priorization_Asset_Index'] = range(len(df))
        df.loc[orden, 'Group_Priorization_Asset_Index'] = range(len(df))
        # print(service_name)
        # print(df['Category_Group_Priorization_Asset_Index'].value_counts())
        df = hi.map_indicator_to_health_using_logistic_function(
            df,
            'Group_Priorization_Asset_Index',
            'standard',
            False,
            'General_Priorization_Asset'
        )
        df = hi.get_partial_health_by_group(
            df,
            'Group_Priorization_Asset_Index',
            'General_Priorization_Asset_Indicator',
            '# de Tienda',
            'General_Priorization_Store_Indicator'
        )
        df = hi.get_partial_health_by_group(
            df,
            'General_Priorization_Asset_Indicator',
            'General_Priorization_Asset_Indicator',
            'Estado/Provincia',
            'General_Priorization_State_Indicator'
        )
        self.logger.info('Success Execution of get_priorization_by_request_class_group method in priorization.py')
        return df
