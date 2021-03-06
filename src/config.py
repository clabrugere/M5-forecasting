import os
from pathlib2 import Path


DIR_ROOT = Path(os.path.abspath(__file__)).parents[1]
DIR_DATA = os.path.join(DIR_ROOT, 'data/')
DIR_DATA_RAW = os.path.join(DIR_DATA, 'raw/')
DIR_DATA_PROCESSED = os.path.join(DIR_DATA, 'processed/')
DIR_DATA_SUBMISSION = os.path.join(DIR_DATA, 'submission/')

DATA_FILES = {
    'calendar': os.path.join(DIR_DATA_RAW, 'calendar.csv'),
    'sales': os.path.join(DIR_DATA_RAW, 'sales_train_validation.csv'),
    'prices': os.path.join(DIR_DATA_RAW, 'sell_prices.csv'),
    'submission': os.path.join(DIR_DATA_RAW, 'sample_submission.csv'),
}


STORES = [
    'CA_1',
    'CA_2',
    'CA_3',
    'CA_4',
    'TX_1',
    'TX_2',
    'TX_3',
    'WI_1',
    'WI_2',
    'WI_3'
]

DEPTS = [
    'HOBBIES_1',
    'HOBBIES_2',
    'HOUSEHOLD_1',
    'HOUSEHOLD_2',
    'FOODS_1',
    'FOODS_2',
    'FOODS_3'
]

CATS = [
    'HOBBIES',
    'HOUSEHOLD',
    'FOODS'
]

STATES = [
    'CA',
    'TX',
    'WI'
]


def get_raw_filename(name):
    return os.path.join(DIR_DATA_RAW, name)


def get_processed_filename(name):
    return os.path.join(DIR_DATA_PROCESSED, name)


def get_submission_filename(name):
    return os.path.join(DIR_DATA_SUBMISSION, name)