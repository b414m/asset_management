import json

class MainConfig:
    def __init__(self, main_configFile: str, logger):
        self.main_configFile = main_configFile
        self.logger = logger

    def get_config(self):
        self.logger.info('Beginning Execution of get_config method in main_config.py')
        main_configuration_execution_f = open('config/' + self.main_configFile, encoding='utf-8')
        main_configuration_execution = json.load(main_configuration_execution_f)
        self.logger.info('Finishing Execution of get_config method in main_config.py')
        return main_configuration_execution
