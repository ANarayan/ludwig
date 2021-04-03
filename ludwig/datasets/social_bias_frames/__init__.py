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
from ludwig.datasets.mixins.download import TarDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import *


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True):
    dataset = SocialBiasFrames(cache_dir=cache_dir)
    return dataset.load(split=split)


class SocialBiasFrames(
    TarDownloadMixin, MultifileJoinProcessMixin, CSVLoadMixin, BaseDataset
):
    """
    Social Bias Frames Dataset
    Details:

    Dataset source:
        Sap, Maarten, et al. "Social bias frames: Reasoning about social
        and power implications of language."
        arXiv preprint arXiv:1911.03891 (2019).
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(
            dataset_name="social_bias_frames", cache_dir=cache_dir
        )

    def load_processed_dataset(self, split):
        processed_df = super(SocialBiasFrames, self).load_processed_dataset(
            split=split
        )
        processed_df["intentYN"] = processed_df["intentYN"].astype(str)
        processed_df["sexYN"] = processed_df["sexYN"].astype(str)
        processed_df["offensiveYN"] = processed_df["offensiveYN"].astype(str)
        processed_df["speakerMinorityYN"] = processed_df["speakerMinorityYN"].astype(str)
        return processed_df
