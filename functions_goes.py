import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from spacepy import pycdf
import glob
import urllib.request
from urllib.request import urlopen
import os.path
import pickle
from bs4 import BeautifulSoup

from tqdm import tqdm

import data_frame_transforms as data_transform
import position_frame_transforms as pos_transform
import functions_general as fgen

from functions_general import load_path

"""
GOES DATA PATH
"""

goes_path = load_path(path_name='goes_path')
print(f"GOES data path loaded: {goes_path}")