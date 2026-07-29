import bisect
import hashlib
import logging
import os
import random
from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

DATA_DIR = join(BASE_DIR, '../data')
SMALL_ATTR_PATH = join(DATA_DIR, 'items_ins_v2_1000.json')
SMALL_FILE_PATH = join(DATA_DIR, 'items_shuffle_1000.json')
FULL_ATTR_PATH = join(DATA_DIR, 'items_ins_v2.json')
FULL_FILE_PATH = join(DATA_DIR, 'items_shuffle.json')

WEBSHOP_DATASET = os.getenv('WEBSHOP_DATASET', 'small').strip().lower()
WEBSHOP_ATTR_DATASET = os.getenv('WEBSHOP_ATTR_DATASET', 'small').strip().lower()
if WEBSHOP_DATASET in {'all', 'full', 'large'}:
    _default_file_path = FULL_FILE_PATH
elif WEBSHOP_DATASET in {'small', '1k', '1000'}:
    _default_file_path = SMALL_FILE_PATH
else:
    raise ValueError(
        "WEBSHOP_DATASET must be one of: small, 1k, 1000, all, full, large"
    )

if WEBSHOP_ATTR_DATASET in {'all', 'full', 'large'}:
    _default_attr_path = FULL_ATTR_PATH
elif WEBSHOP_ATTR_DATASET in {'small', '1k', '1000'}:
    _default_attr_path = SMALL_ATTR_PATH
else:
    raise ValueError(
        "WEBSHOP_ATTR_DATASET must be one of: small, 1k, 1000, all, full, large"
    )

DEFAULT_ATTR_PATH = os.getenv('WEBSHOP_ATTR_PATH', _default_attr_path)
DEFAULT_FILE_PATH = os.getenv('WEBSHOP_FILE_PATH', _default_file_path)
DEFAULT_REVIEW_PATH = join(BASE_DIR, '../data/reviews.json')

FEAT_CONV = join(BASE_DIR, '../data/feat_conv.pt')
FEAT_IDS = join(BASE_DIR, '../data/feat_ids.pt')

HUMAN_ATTR_PATH = join(BASE_DIR, '../data/items_human_ins.json')
HUMAN_ATTR_PATH = join(BASE_DIR, '../data/items_human_ins.json')

def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx

def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()
