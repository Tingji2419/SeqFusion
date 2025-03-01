# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import logging
import os
import pathlib
import sys
from urllib import request


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = '../academic_data/m4') -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        train_cache_file = os.path.join(dataset_file, 'training.npz')
        test_cache_file = os.path.join(dataset_file, 'test.npz')
        m4_info = pd.read_csv(info_file)
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             train_cache_file if training else test_cache_file,
                             allow_pickle=True))


@dataclass()
class M4Meta:
    seasonal_patterns = ['yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly']
    horizons_map = {
        'yearly': 6,
        'quarterly': 8,
        'monthly': 18,
        'weekly': 13,
        'daily': 14,
        'hourly': 48
    }  # different predict length
    lookback_window_map = {
        'yearly': 9,
        'quarterly': 16,
        'monthly': 36,
        'weekly': 65,
        'daily': 9,
        'hourly': 2
    }
    seasonal_map = {
        'yearly': 'yearly',
        'quarterly': 'quarterly',
        'monthly': 'monthly',
        'weekly': 'monthly',
        'daily': 'monthly',
        'hourly': 'monthly'
    }
    # history_size = {
    #     'yearly': 1.5,
    #     'quarterly': 1.5,
    #     'monthly': 1.5,
    #     'weekly': 10,
    #     'daily': 10,
    #     'hourly': 10
    # }  # from interpretable.gin


def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)


@dataclass()
class M3Meta:
    seasonal_patterns = ['yearly', 'quarterly', 'monthly', 'other']
    horizons_map = {
        'yearly': 6,
        'quarterly': 8,
        'monthly': 18,
        'other': 8,
    }  # different predict length
    lookback_window_map = {
        'yearly': 12,
        'quarterly': 24,
        'monthly': 24,
        'other' : 16
    }
    seasonal_map = {
        'yearly': 'yearly',
        'quarterly': 'quarterly',
        'monthly': 'monthly',
        'other': 'monthly',
    }
    # history_size = {
    #     'yearly': 1.5,
    #     'quarterly': 1.5,
    #     'monthly': 1.5,
    #     'other': 1.5,
    # }

@dataclass()
class TourismMeta:
    seasonal_patterns = ['yearly', 'quarterly', 'monthly']
    horizons_map = {
        'yearly': 4,
        'quarterly': 8,
        'monthly': 24,
    }  # different predict length
    lookback_window_map = {
        'yearly': 12,
        'quarterly': 24,
        'monthly': 36,
    }
    seasonal_map = {
        'yearly': 'yearly',
        'quarterly': 'quarterly',
        'monthly': 'monthly',
    }
    # history_size = {
    #     'yearly': 1.5,
    #     'quarterly': 1.5,
    #     'monthly': 1.5,
    # }