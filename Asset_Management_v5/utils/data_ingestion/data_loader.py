import pandas as pd
import keyring
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text, MetaData
from databricks import sql
import json

class DataLoader:
    """
    A class for loading data from various sources
    """
    def __init__(self, connection, logger):
        """
        Initializes the DataLoader class.
        """
        global target_table, db
        self.connection = connection
        self.engine = connection.get_engine()
        self.logger = logger
        # Database configuration pulled from connection object
        target_table = connection.db_config['target_table']
        db = connection.db_config.get('db', None)

    def get_data_table(self):
        self.logger.info('Beginning Execution of get_data_table method in data_loader.py')
        full_table_name = f"{db}.dbo.{target_table}"
        query = f"SELECT * FROM {full_table_name}"
        self.logger.info('Success Execution of get_data_table method in data_loader.py')
        return pd.read_sql(query, self.engine)

    def get_tables_names_from_db(self) -> pd.DataFrame:
        """
        Placeholder for method to retrieve table names or metadata from TRIRIGA.
        Returns:
          pd.DataFrame: DataFrame with table names or other metadata
        """
        # TODO: Implement retrieval of table names or metadata
        raise NotImplementedError("get_tables_names_from_db not implemented yet.")

    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Loads data from the specified file type.
        Supported types: 'csv', 'excel', 'json', 'sql-tririga', 'databricks-lat'
        """
        self.logger.info('Beginning Execution of load_data method in data_loader.py')
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'excel':
            data = pd.read_excel(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        elif file_type == 'sql-tririga':
            data = self.get_data_table()
        elif file_type == 'databricks-lat':
            data = self.get_lat_data()
        else:
            raise ValueError("Supported files are the following: csv, excel, json, sql-tririga, databricks-lat")
        self.logger.info('Success Execution of load_data method in data_loader.py')
        return data

    def get_lat_connection_parameters(self):
        self.logger.info('Beginning Execution of get_lat_connection_parameters method in data_loader.py')
        with open('utils/data_ingestion/connection_config/connection_params_databricks.json') as f:
            db_config = json.load(f)
        host = db_config['HOST']
        http_path = db_config['HTTP_PATH']
        token = db_config['TOKEN']
        cluster_id = db_config['CLUSTER_ID']
        warehouse_id = db_config['WAREHOUSE_ID']
        self.logger.info('Success Execution of get_lat_connection_parameters method in data_loader.py')
        return host, http_path, token, cluster_id, warehouse_id

    def get_lat_data(self):
        self.logger.info('Beginning Execution of get_lat_data method in data_loader.py')
        host, http_path, token, _, _ = self.get_lat_connection_parameters()
        query = "SELECT * FROM silver_prod_lat.silver_dd_lat_operating_store"
        with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                cols = [col[0] for col in cursor.description]
        df = pd.DataFrame(result, columns=cols)
        self.logger.info('Success Execution of get_lat_data method in data_loader.py')
        return df

    def get_all_assets_data(self, file_path_work_orders: str, file_path_assets: str) -> pd.DataFrame:
        """
        Loads data available on sharepoint path: Intensity 3.0/Documents/Bases de Datos
        Args:
            file_path_work_orders (str): Path to the Work Orders file.
            file_path_assets (str, optional): Path to the Assets file.
        Returns:
            pd.DataFrame: return the union of DataFrames with work orders and assets data.
                         "Asset and Work Task Data - Full Dataset.csv" and "Asset and Work Task Data - Full Dataset_10172024.csv" files.
        """
        self.logger.info('Beginning Execution of get_all_assets_data method in data_loader.py')
        df_assets_without_work_orders = self.load_data(file_path_work_orders)
        df_work_orders = self.load_data(file_path_assets)
        df_all_assets = pd.concat([df_assets_without_work_orders, df_work_orders])
        self.logger.info('Success Execution of get_all_assets_data method in data_loader.py')
        return df_all_assets
