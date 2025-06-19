import argparse
import json
from utils.general_priorization_analysis import GeneralPriorizationAnalysis
from utils.data_ingestion.data_uploader import DataUploader
from utils.logger import Logger
from utils.main_config import MainConfig

logger = Logger().get_logger(show_in_console=True)
dul = DataUploader(logger)

def parse_args():
    logger.info('Beginning Execution of parse_args method in main.py')
    parser = argparse.ArgumentParser(description="General Priorization of Asset Management")
    parser.add_argument(
        '--save-data',
        action="store_true",
        help="Indicates if General Priorization of Asset Management report will be stored in SQL Server defined in connection_params_non_prod.json file"
    )
    parser.add_argument(
        "--config_file",
        "-config",
        required=True,
        help="name of main configuration file contained in config directory"
    )
    logger.info('Success Execution of parse_args method in main.py')
    return parser.parse_args()

def main():
    logger.info('Begining execution of pipeline')
    report_path = 'outputs/datasets/general_report_servicesVSigmode'
    args = parse_args()
    main_configFile = args.config_file
    logger.info(main_configFile)
    main_configuration_execution = MainConfig(main_configFile, logger).get_config()
    general_report = GeneralPriorizationAnalysis(main_configuration_execution, logger)
    general_report_services = general_report.run()
    general_report_services.to_excel(report_path + '.xlsx')
    general_report_services.to_pickle(report_path + '.pkl')
    if args.save_data:
        dul.test_upload_data(report_path + '.pkl', if_exists='replace', save_copy=True)
    logger.info('Success execution of pipeline')

if __name__ == '__main__':
    main()
