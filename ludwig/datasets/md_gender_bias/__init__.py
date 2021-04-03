#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import  TarDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import *

def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True):
    dataset = MDGenderBias(cache_dir=cache_dir)
    return dataset.load(split=split)

class MDGenderBias(TarDownloadMixin, MultifileJoinProcessMixin,
                 CSVLoadMixin, BaseDataset):
    """
        MD Gender Bias Dataset
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, task='wizard'):
        super().__init__(dataset_name="md_gender_bias", cache_dir=cache_dir)
        self.task = task

    @property
    def download_filenames(self):
        return self.config['split_filenames'][self.task]

    @property
    def processed_dataset_path(self):
        return os.path.join(self.download_dir, 'processed', self.task)

    @property
    def processed_temp_path(self):
        return os.path.join(self.download_dir, '_processed', self.task)


    
