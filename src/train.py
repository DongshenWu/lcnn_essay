#!/usr/bin/env python3
import logging
import os
from argparse import ArgumentParser
from time import strftime, gmtime

from parsers import SettingParser
from trainer import Train


def parse_arguments():
    parser = ArgumentParser('Train one model from a settings.yml.')
    parser.add_argument('root_path', type=str, help='Run directory.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--jobid', type=str, default='', help='Slurm job id; suffixes the run dir.')
    parser.add_argument('--save-memory', action='store_true', help='Skip writing model_state_dict.pth.')
    parser.add_argument('--debug', action='store_true', help='1-epoch run with VerboseModel instrumentation.')
    return parser.parse_args()


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(
        '%(asctime)-8s %(message)s', datefmt='%Y%m%d:%H:%M:%S'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_run_id(job_id: str = ""):
    suffix = ('.' + job_id) if job_id else ''
    return strftime("%Y%m%d_%H%M%S", gmtime()) + suffix


def init_run(run_id: str, root_path, device: str):
    run_path = root_path + '/' + run_id
    os.mkdir(run_path)
    logger = setup_logger(name=run_id, log_file=run_path + '/training.log')
    logger.info('Loading of the setting file...')
    setting_parser = SettingParser(root_path + '/settings.yml')
    logger.info(f'Running on device {device}')
    setting_parser.update_setting(f'device: {device}')
    logger.info('Saving current configuration...')
    setting_parser.save_setting(run_path + '/dumped_setting.yml')
    logger.info(str(setting_parser))
    logger.info('Done.')
    return setting_parser, run_path, logger


def start_train():
    parsed_args = parse_arguments()
    main_logger = setup_logger(
        name='main', log_file=parsed_args.root_path + '/single_training.log')
    run_id = get_run_id(parsed_args.jobid)
    if parsed_args.debug:
        run_id += '_DEBUG'
    main_logger.info(f'> Start single training with run id: {run_id}')
    main_logger.info(f'>> Run path: {parsed_args.root_path}')
    main_logger.info(f'>> Device: {parsed_args.device}')
    main_logger.info(f'>> Debug mode: {parsed_args.debug}')
    main_logger.info(f'>> Save memory mode: {parsed_args.save_memory}')
    main_logger.info('')

    setting_parser, run_path, logger = init_run(
        run_id, parsed_args.root_path, parsed_args.device)
    logger.info('Loading objects...')
    try:
        setting_parser.load_setting()
    except Exception as exception:
        logger.info('Error while loading objects:')
        logger.exception(exception)
        return

    if parsed_args.debug:
        logger.info('Debug mode: 1 epoch with VerboseModel.')
        from utils.inspections import VerboseModel
        setting_parser.model = VerboseModel(setting_parser.model)
        setting_parser.update_setting('epochs: 1')

    logger.info('Initializing trainer...')
    trainer = Train(**setting_parser.to_dict(), logger=logger)
    logger.info('Start training...')
    try:
        trainer.run(run_path, save_state_dict=not parsed_args.save_memory)
    except Exception as exception:
        logger.info('Error while training:')
        logger.exception(exception)


if __name__ == '__main__':
    start_train()
