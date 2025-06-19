import pandas as pd
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
from urllib.parse import quote_plus
import json
import keyring

class SQLDataConnection:
    """A class to generate connection to the TRIRIGA SQL-Server DB"""
    def __init__(self, db_configFile: str, logger):
        """
        Initializes the SQLDataConnection class.
        Args:
            db_configFile (str): Path to the connectionParams file, which contains: server, db, driver, table data.
            passw (str): Password associated to user registered with keyring for server connection
        """
        self.db_configFile = db_configFile
        self.logger = logger
        f = open('utils/data_ingestion/connection_config/' + self.db_configFile)
        self.db_config = json.load(f)
        self.userN = self.db_config['user']
        self.keyref = self.db_config['keyring_ref']
        self.passw = keyring.get_password(self.keyref, self.userN)
        self.db = self.db_config['db']
        self.server = self.db_config['server']

    def get_engine(self):
        """
        Get the TRIRIGA data in a pd.DataFrame format.
        Returns:
          pd.DataFrame: Dataframe with TRIRIGA data from defined query
        """
        self.logger.info('Beginning Execution of get_engine method in sql_data_connection.py')
        self.db_config['password_quote'] = quote_plus(self.passw)
        connection_string = '''mssql+pyodbc://{0};{1}@{2}/{3}?driver={4}&Authentication=ActiveDirectoryPassword'''.format(
            self.userN,
            self.db_config['password_quote'],
            self.db_config['server'],
            self.db_config['db'],
            self.db_config['driver']
        )
        engine = create_engine(connection_string, poolclass=NullPool)
        try:
            engine.connect()
            self.logger.info(f"Success Connection for server:{self.db_config['server']}, db:{self.db_config['db']}")
        except Exception as e:
            self.logger.info(e)
        self.logger.info('Success Execution of get_engine method in sql_data_connection.py')
        return engine
