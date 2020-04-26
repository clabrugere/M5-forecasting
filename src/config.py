import os
from pathlib2 import Path


DIR_ROOT = Path(os.path.abspath(__file__)).parents[1]
DIR_DATA = os.path.join(DIR_ROOT, 'data/')
DIR_DATA_RAW = os.path.join(DIR_DATA, 'raw/')
DIR_DATA_PROCESSED = os.path.join(DIR_DATA, 'processed/')

DATA_FILES = {
    'calendar': os.path.join(DIR_DATA_RAW, 'calendar.csv'),
    'sales': os.path.join(DIR_DATA_RAW, 'sales_train_validation.csv'),
    'prices': os.path.join(DIR_DATA_RAW, 'sell_prices.csv'),
    'submission': os.path.join(DIR_DATA_RAW, 'sample_submission.csv'),
}


def get_raw_filename(name):
    return os.path.join(DIR_DATA_RAW, name)


def get_processed_filename(name):
    return os.path.join(DIR_DATA_PROCESSED, name)
