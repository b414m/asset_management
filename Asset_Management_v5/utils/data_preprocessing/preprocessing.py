import pandas as pd import numpy as np import unicodedata import re

class Preprocessing: 
    def init(self, main_configuration_execution, logger): 
        global valid_store_formats, to_status, status_to_valid_values, partial_cost, total_savings,
        responsable_organization_sap_id, start_date_execution_to, end_date_execution_to, granularity_report_type 
        self.logger = logger 
        self.main_configuration_execution = main_configuration_execution 
        valid_store_formats = self.main_configuration_execution['valid_store_formats'] 
        to_status = self.main_configuration_execution['fields_input_assets_requirements']['to_status'] 
        status_to_valid_values = self.main_configuration_execution['status_to_valid_values'] 
        partial_cost = self.main_configuration_execution['fields_input_assets_requirements']['partial_cost'] 
        total_savings = self.main_configuration_execution['fields_input_assets_requirements']['total_savings'] 
        responsable_organization_sap_id = self.main_configuration_execution['fields_input_assets_requirements']['responsable_organization_sap_id'] 
        start_date_execution_to = self.main_configuration_execution['fields_input_assets_requirements']['start_date_execution_to'] 
        end_date_execution_to = self.main_configuration_execution['fields_input_assets_requirements']['end_date_execution_to'] 
        granularity_report_type = self.main_configuration_execution['granularity_report_type']
    
    def preprocess_cost(self, pre_df):
        df = pre_df.copy()
        df[partial_cost] = df[partial_cost].str.replace('.', '').astype(float)
        df[total_savings] = df[total_savings].str.replace('.', '').astype(float)
        df['Costo'] = np.where(df[responsable_organization_sap_id].isna(), df[total_savings], df[partial_cost])
        return df
    
    def preprocess_reparation_time(self, pre_df):
        df = pre_df.copy()
        df['reparation_time'] = df[end_date_execution_to] - df[start_date_execution_to]
        return df
    
    def rename_output_fields(self, pre_df):
        df = pre_df.copy()
        relations_column_names = pd.read_excel('data_sources/relations_column_names.xlsx')
        relations_column_names = relations_column_names[
            (relations_column_names['flag'] == True) & 
            (relations_column_names['dataset'] == 'indicadores')
        ]
        columns_renamed = relations_column_names.set_index('field')['rename'].to_dict()
        filter_columns = relations_column_names['rename'].dropna().to_list()
        df = df.rename(columns=columns_renamed)
        df = df[filter_columns]
        return df

    def filter_to_valid_status(self, pre_df):
        df = pre_df.copy()
        df = df[df[to_status].isin(status_to_valid_values)]
        return df

    def general_preprocessing(self, pre_df):
        self.logger.info('Beginning Execution of general_preprocessing method in preprocessing.py')
        df = pre_df.copy()
        # fileMapping = 'data_sources/Mapping_TRIRIGA_ASSET.xlsx'
        operations_dict = {
            'Costo': ['sum', 'mean'],
            'reparation_time': ['sum', 'mean']
        }
        groups_operations = [granularity_report_type]

        # Refactorización del código, para que el método de preprocesamiento de grupos
        # sea agnóstico independientemente de la taxonomía de la categoría
        # dejar el método preprocessing_groups y dentro de él definir cada categoría de preprocesamiento
        # BRIDGE (patrón de diseño)

        # df = self.mapping_fields_assets_to_tririga(fileMapping, df)
        df = self.filter_valid_formats(df)
        df = self.filter_valid_subsystems(df)
        df = self.filter_to_valid_status(df)
        df = self.preprocess_cost(df)
        df = self.preprocess_reparation_time(df)
        df = self.preprocessing_rule_of_filter_from_analysis(df)
        df = self.preprocessing_dates(df)
        df = self.preprocessing_states(df)
        df = self.formatting_assets(df)
        df = self.apply_operations_by_list_and_group(df, groups_operations, operations_dict)
        self.logger.info('Success Execution of general_preprocessing method in preprocessing.py')
        return df
    
    def filter_valid_subsystems(self, pre_df):
        self.logger.info('Beginning Execution of filter_valid_subsystems method in preprocessing.py')
        df = pre_df.copy()
        importance_df = pd.read_excel('data_sources/service_importance.xlsx')
        valid_subsystems = importance_df[importance_df['Categoria Importancia Definida por Negocio'] != 'N/A']['Subsistema'].tolist()
        df = df[df['Pre-Service'].isin(valid_subsystems)]
        self.logger.info('Success Execution of filter_valid_subsystems method in preprocessing.py')
        return df

    def mapping_fields_assets_to_tririga(self, file_path_mapping: str, pre_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map the column names of the Assets and Work Orders Data in TRIRIGA names using file_path_mapping as reference.
        Args:
            file_path_mapping (str): path of mapping file
            pre_df (pd.DataFrame): the dataframe to map column names in to TRIRIGA columns, both must be referenced in the mapping file
        Returns:
            pd.DataFrame: dataframe with columns renamed to TRIRIGA, task type values translated to Spanish, and a 'Descripción' column added if missing.
        """
        self.logger.info('Beginning Execution of mapping_fields_assets_to_tririga method in preprocessing.py')
        df_mapping = pd.read_excel(file_path_mapping)
        mapping = df_mapping[['TRIRIGA','Asset']].set_index('Asset')['TRIRIGA'].to_dict()
        df = pre_df.copy()
        df = df.rename(columns=mapping)
        df['Tipo de Tarea'] = df['Tipo de Tarea'].replace({'Corrective':'Correctivo','Preventive':'Preventivo'})
        if 'Descripción' not in df.columns:
            df['Descripción'] = ''
        self.logger.info('Success Execution of mapping_fields_assets_to_tririga method in preprocessing.py')
        return df

    def preprocessing_request_class(self, pre_df, request_class_field):
        self.logger.info('Beginning Execution of preprocessing_request_class method in preprocessing.py')
        df = pre_df.copy()
        df['request_class_preproc'] = df[request_class_field].apply(lambda x: str(x).lower())
        df['request_class_preproc'] = df['request_class_preproc'].apply(lambda x: unicodedata.normalize('NFKD', x))
        df['request_class_preproc'] = df['request_class_preproc'].apply(lambda x: ''.join(char for char in x if not unicodedata.combining(char)))
        df['request_class_preproc'] = df['request_class_preproc'].apply(lambda x: re.sub(r'[^a-z0-9 ]', '', x))
        df['request_class_preproc'] = df['request_class_preproc'].apply(lambda x: re.sub(r'\s+', ' ', x))
        df['request_class_preproc'] = df['request_class_preproc'].str.strip()
        self.logger.info('Success Execution of preprocessing_request_class method in preprocessing.py')
        return df
    
    def preprocessing_rule_of_filter_from_analysis(self, pre_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Beginning Execution of preprocessing_rule_of_filter_from_analysis method in preprocessing.py')
        df = pre_df.copy()
        exceptions_termino_real = ['6423-11-29']
        # We define the next fields in order to make a data observability for data governance audit
        df['data_comments'] = ''
        df['field_implied_in_data_comments'] = ''
        # 'data_comments_years_since_last_touch_1',
        # 'field_comments_years_since_last_touch',
        column_comments = [
            'data_comments_termino_real_1', 'data_comments_termino_real_2',
            'data_comments_working_hours_1', 'data_comments_working_hours_2',
            'data_comments_costo_total_1', 'data_comments_estado_provincia_1',
            'data_comments_id_del_activo_1', 'data_comments_numero_tienda_1'
        ]

        fields_implied_columns = [
            'field_comments_termino_real', 'field_comments_working_hours',
            'field_comments_costo_total', 'field_comments_estado_provincia',
            'field_comments_id_del_activo', 'field_comments_numero_tienda'
        ]

        # Rules of years_since_last_touch:
        # df['data_comments_years_since_last_touch_1'] = np.where(df['years_since_last_touch'].isna(),
        #     'The "years_since_last_touch" value is a NaN value', '')
        # df['field_comments_years_since_last_touch'] = np.where(
        #     df['data_comments_years_since_last_touch_1'] != '', '# de Tienda', '')

        # Rule of Termino Real:
        df['data_comments_termino_real_1'] = np.where(
            df['Termino Real'].astype(str).isin(exceptions_termino_real),
            '', 'The date value in field "Termino Real" is not valid'
        )
        df['data_comments_termino_real_2'] = np.where(
            df['Termino Real'].isna(),
            'The "Termino Real" value is a NaN value', ''
        )
        df['field_comments_termino_real'] = np.where(
            (df['data_comments_termino_real_1'] != '') |
            (df['data_comments_termino_real_2'] != ''),
            'Termino Real', ''
        )

        # Rule of current working hours:
        outliers_tot_curr_wor_hr = df[
            (df['Tot Current Working Hours'] < 0) |
            (df['Tot Current Working Hours'] > 50000)
        ]['Tot Current Working Hours'].to_list()
        df['data_comments_working_hours_1'] = np.where(
            df['Tot Current Working Hours'].isin(outliers_tot_curr_wor_hr),
            'The Tot Current Working Hours is an outlier value', ''
        )
        df['data_comments_working_hours_2'] = np.where(
            df['Tot Current Working Hours'].isna(),
            'The Tot Current Working Hours is a NaN value', ''
        )
        df['field_comments_working_hours'] = np.where(
            (df['data_comments_working_hours_1'] != '') |
            (df['data_comments_working_hours_2'] != ''),
            'Tot Current Working Hours', ''
        )

        # Rule of Costo Total de Proveedor:
        df['data_comments_costo_total_1'] = np.where(
            df['Costo Total de Proveedor'] <= 0,
            'The value of Costo Total de Proveedor is not greater than zero', ''
        )
        df['field_comments_costo_total'] = np.where(
            df['data_comments_costo_total_1'] != '',
            'Costo Total de Proveedor', ''
        )

        # Rule of Costo Estado/Provincia:
        df['data_comments_estado_provincia_1'] = np.where(
            df['Estado/Provincia'].isna(),
            'The Estado/Provincia is a NaN value', ''
        )

        df['field_comments_estado_provincia'] = np.where(
        df['data_comments_estado_provincia_1'] != '',
        'Estado/Provincia',
        ''
        )

        # Rule of ID del activo:
        df['data_comments_id_del_activo_1'] = np.where(
            df['ID del activo'] == '',
            'The value of ID del activo is empty',
            ''
        )
        df['field_comments_id_del_activo'] = np.where(
            df['data_comments_id_del_activo_1'] != '',
            'ID del activo',
            ''
        )

        # Rule of Store Number:
        df['data_comments_numero_tienda_1'] = np.where(
            df['# de Tienda'].isna(),
            'The value of Store Number is empty',
            ''
        )
        df['field_comments_numero_tienda'] = np.where(
            df['data_comments_numero_tienda_1'] != '',
            '# de Tienda',
            ''
        )

        # Integrating and Formatting Report
        df['data_comments'] = df[column_comments].apply(
            lambda row: '|'.join(row.values.astype(str)),
            axis=1
        )
        df['data_comments'] = df['data_comments'].str.replace(r'{2,}', ' ', regex=True)
        df['data_comments'] = df['data_comments'].str.replace(r'\s+', ' ', regex=True)
        df['data_comments'] = df['data_comments'].str.replace(r'\s+', ' ', regex=True)
        df['data_comments'] = df['data_comments'].replace(' | N/A', '')

        df['field_implied_in_data_comments'] = df[fields_implied_columns].apply(
            lambda row: ','.join(row.values.astype(str)),
            axis=1
        )
        df['field_implied_in_data_comments'] = df['field_implied_in_data_comments'].str.replace(r'{2,}', ' ', regex=True)
        df['field_implied_in_data_comments'] = df['field_implied_in_data_comments'].str.replace(r'^,', '', regex=True)
        df['field_implied_in_data_comments'] = df['field_implied_in_data_comments'].str.replace(r',$', '', regex=True)
        df['field_implied_in_data_comments'] = df['field_implied_in_data_comments'].replace(' | N/A', '')

        df = df.drop(columns=column_comments + fields_implied_columns)
        self.logger.info('Success Execution of preprocessing_rule_of_filter_from_analysis method in preprocessing.py')
        return df

    def preprocessing_dates(self, pre_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the inexistent dates and Create the columns Termino Real año and Termino Real mes using the column Termino Real
        Args:
            df (pd.DataFrame): Dataframe to apply the preprocessing
        Returns:
            pd.DataFrame: Dataframe with the steps mentioned above
        """
        self.logger.info('Beginning Execution of preprocessing_dates method in preprocessing.py')
        #exceptions_termino_real = ['6423-11-29']
        df = pre_df.copy()
        #df = df[~df['Termino Real'].astype(str).isin(exceptions_termino_real)]
    
        df['Termino Real año'] = pd.DatetimeIndex(df['Termino Real']).year
        df['Termino Real mes'] = pd.DatetimeIndex(df['Termino Real']).month
        df['Termino Real'] = df['Termino Real'].astype('datetime64[ns]')
        df = df.sort_values('Termino Real')
        self.logger.info('Success Execution of preprocessing_dates method in preprocessing.py')
        return df
    
    def preprocessing_down_time(self, pre_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Beginning Execution of preprocessing_dates method in preprocessing.py')
        df = pre_df.copy()
        outliers_tot_curr_wor_hr = df[
            (df['Tot Current Working Hours'] < 0) |
            (df['Tot Current Working Hours'] > 50000)
        ]['Tot Current Working Hours'].to_list()
        df = df[~df['Tot Current Working Hours'].isin(outliers_tot_curr_wor_hr)]
        df = df[df['Tot Current Working Hours'].isna()]
        self.logger.info('Success Execution of preprocessing_dates method in preprocessing.py')
        return df

    def text_normalized(self, pre_df, text_field):
        self.logger.info('Beginning Execution of text_normalized method in preprocessing.py')
        df = pre_df.copy()
        df[text_field + '_source'] = df[text_field]
        # convert tu minus and remove accents
        df[text_field] = df[text_field].astype(str).apply(
            lambda x: None if x is None else unicodedata.normalize('NFKD', x.lower())
                                             .encode('ASCII', 'ignore')
                                             .decode('utf-8')
        )
        # remove special characters
        df[text_field] = df[text_field].astype(str).apply(
            lambda x: None if x is None else re.sub(r'[^a-z\s]', '', x)
        )
        # remove consecutive duplid characters
        df[text_field] = df[text_field].astype(str).apply(
            lambda x: None if x is None else re.sub(r'(.)\1+', r'\1', x)
        )
        # remove space
        df[text_field] = df[text_field].astype(str).apply(
            lambda x: None if x is None else x.replace(' ', '')
        )
        self.logger.info('Success Execution of text_normalized method in preprocessing.py')
        return df

    def preprocessing_states(self, pre_df):
        self.logger.info('Beginning Execution of preprocessing_states method in preprocessing.py')
        state_column = 'Estado/Provincia'
        df = pre_df.copy()
        df = self.text_normalized(df, state_column)
        states = pd.read_excel('data_sources/catalogo_estados_normalizados.xlsx')
        states_normalized_dict = states.set_index('NOMBRE_ENTIDAD_normalized')['NOMBRE_ENTIDAD'].to_dict()
        df[state_column] = df[state_column].replace(states_normalized_dict)
        self.logger.info('Success Execution of preprocessing_states method in preprocessing.py')
        return df

    def formating_assets(self, pre_df: pd.DataFrame) -> pd.DataFrame:
        """
        Include the next steps:
        a) Fill with 0 the NaN values and filter values greater than zero of column: 'Costo Total de Proveedor'
        b) Remove records without value in column: 'ID del activo'
        c) Convert String to float, removing ',' characters in the values of column: 'Costo Total de Proveedor'
        d) Remove records without value in column: 'Termino Real'
        e) Convert String to datetime in format yy/mm/dd the column: 'Termino Real'
        f) Convert to string format the column: '# de Tienda'
        g) Fill with '' the NaN values of column: 'Description'
        Args:
            df (pd.DataFrame) the DataFrame to apply tje steps mentioned above
        Return:
            pd.DataFRame: a DataFrame with formated data
       """
        self.logger.info('Beginning Execution of formating_assets method in preprocessing.py')
        df = pre_df.copy()
        df['Costo Total de Proveedor'] = df['Costo Total de Proveedor'].fillna(0)
        df['Costo Total de Proveedor'] = df['Costo Total de Proveedor'].apply(lambda x: float(str(x).replace(',', '')))
        #df['ID del activo'] = df['ID del activo'].fillna('')
        df['# de Tienda'] = df['# de Tienda'].astype(str)
        self.logger.info('Success Execution of formating_assets method in preprocessing.py')
        return df

    def preprocessing_groups_refrigeration_assets(self, pre_df: pd.DataFrame) -> pd.DataFrame:
        """
        In case we focused on refrigeration assets, this method resume groups using the column: 'Nombre de Activo'
        Args:
            df (pd.DataFrame): Dataframe to be preprocessed

        Returns:
            pd.DataFrame: Dataframe adding the column: Grupo_Nombre using the column 'Nombre de Activo' with the name of group of asset.
        """
        self.logger.info('Beginning Execution of formating_assets method in preprocessing.py')
        df = pre_df.copy()
        df['Grupo_Nombre'] = df['Nombre de Activo'].apply(
            lambda x: x
                .replace('Refrigeracion - ', 'Refrigeracion ')
                .replace('- Temperatura Media', '')
                .replace('- Temperatura Baja', '')
                .replace('- Media Temperatura', '')
                .replace('Media Temperatura', '')
                .replace('Temperatura Media', '')
                .replace('Baja Temperatura', '')
                .replace('Temperatura Baja', '')
                .replace('Temperatura Mixta', '')
                .replace(' ', '')
                .replace('Refrigeracion', 'Refrigeracion')
        )
        df['Grupo_Nombre'] = df['Grupo_Nombre'].apply(
            lambda x: '-'.join(x.split('-')[1:]).split('-')[0].lower() if len(x.split('-')) > 1 else x
        )
        df['Grupo_Nombre'] = (
            df['Grupo_Nombre']
              .str.replace(r'.*camaras.*', 'camaras', regex=True)
              .replace(r'.*bunkers.*', 'bunkers', regex=True)
              .replace(r'.*autocontenidos.*', 'autocontenidos', regex=True)
              .replace(r'.*vitrina.*', 'vitrina', regex=True)
              .replace(r'.*cuarto.*', 'cuarto', regex=True)
              .replace(r'.*reachin.*', 'reach in', regex=True)
              .replace(r'.*rack rack.*', 'rack cabina')
              .replace(r'.*enfriador de agua', 'produccion enfriador de agua')
        )
        self.logger.info('Success Execution of formating_assets method in preprocessing.py')
        return df
    
    def get_list_of_elements_in_column_by_group(self, df: pd.DataFrame, column: str, group: str):
        """
        Get the list of elements for a group in a column target
        Args:
            pre_df (pd.DataFrame): DataFrame to apply the method
            column (str): name of column target
            group (str): name of group target
        Returns:
            __type__: Dataframe with the preprocessed data
        """
        self.logger.info('Beginning Execution of get_list_of_elements_in_column_by_group method in preprocessing.py')
        groupStr = '_'.join([x.replace('#','').replace(' ','') for x in group])
        type_of_column = df[column].dtype
        df[column] = df[column].astype(str)
        df['list_'+column+'_'+group] = df.groupby(group)[column].transform(','.join)
        df[column] = df[column].astype(type_of_column)
        self.logger.info('Success Execution of get_list_of_elements_in_column_by_group method in preprocessing.py')
        return df

    def apply_operations_by_list_and_group(self, pre_df: pd.DataFrame, groups_operations: list, operations_dict: dict):
        """
        Apply the transform functions asociated to columns using specific groups,
        all of these are defined by operations_dict and groups_operations.
        Args:
            pre_df (pd.DataFrame): _description_
            groups_operations (list): list of groups to apply the operations
            operations_dict (dict): dictionary with columns as keys and the values are operations asociated to every column
        Raises:
            ValueError: In case operation are not supported, the operations allowed are the following: 'list','sum','mean','median','std','count','max','min'
        Returns:
            __type__: Return a DataFrame with the preprocessing method
        """
        self.logger.info('Beginning Execution of apply_operations_by_list_and_group method in preprocessing.py')
        valid_operations = ['list','sum','mean','median','std','count','max','min']
        all_operations = [operation for operations in operations_dict.values() for operation in operations]
        df = pre_df.copy()
        if len(list(set(all_operations) - set(valid_operations))) > 0:
            raise ValueError("Supported operations are the following:'list','sum','mean','median','std','count','max','min'")
        for group in groups_operations:
            groupStr = '_'.join([x.replace('#','').replace(' ','') for x in group])
            for column, operations in operations_dict.items():
                for operation in operations:
                    if operation != 'list':
                        df[operation+'_'+column+'_'+group] = df.groupby(group)[column].transform(operation)
                    else:
                        self.get_list_of_elements_in_column_by_group(df, column, group)
        self.logger.info('Success Execution of apply_operations_by_list_and_group method in preprocessing.py')

    def filter_valid_formats(self, pre_df):
        self.logger.info('Beginning Execution of filter_valid_formats method in preprocessing.py')
        df = pre_df.copy()
        df = df[df['Formato de negocio'].isin(valid_store_formats)]
        self.logger.info('Success Execution of filter_valid_formats method in preprocessing.py')
        return df

    def preprocessing_lat_stores(self, pre_df):
        self.logger.info('Beginning Execution of preprocessing_lat_stores method in preprocessing.py')
        df = pre_df.copy()
        df = df[df['Mercado']=='MX']
        df['STORE_NUMBER'] = df['STORE_NUMBER'].astype(int)
        df['FORMAT'] = df['FORMAT'].replace('Sams Club', "Sam's Club")
        df = df[df['FORMAT'].isin(valid_store_formats)]
        self.logger.info('Success Execution of preprocessing_lat_stores method in preprocessing.py')
        return df

    def filter_request_class(self, pre_df):
        self.logger.info('Beginning Execution of filter_request_class method in preprocessing.py')
        df_request_class = pre_df.copy()
        names = pre_df.copy()
        column_id_activo = 'Asset ID'
        col_req_class    = 'Pre-Service'
        col_task_type    = 'Task Type'
        valid_assets     = df_request_class.groupby(column_id_activo)[col_req_class].nunique()
        # valid_assets = valid_assets[valid_assets == 1].index
        # df_request_class = df_request_class[df_request_class[column_id_activo].isin(valid_assets)]
        df_request_class['count_request_class'] = df_request_class.groupby(column_id_activo)[col_req_class].transform('nunique')
        df_request_class = df_request_class[df_request_class['count_request_class'] == 1]
        df_request_class = df_request_class[df_request_class[col_task_type] == 'Corrective']
        names = names[names[col_req_class].isna()]
        names = names.drop_duplicates(subset=[column_id_activo])
        df = pd.concat([df_request_class,names])
        df = df[(df['Asset Status']=='Available')]
        self.logger.info('Sucess Execution of filter_request_class method in preprocessing.py')
        return df

