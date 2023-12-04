import random
import typing

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import audb
import audeer

from . import (
    CATEGORY_LABELS,
    SEED
)

random.seed(SEED)


class RandomGaussian:
    r"""Return predictions from a model that returns Gaussian distributed values
    """

    def __init__(self) -> None:
        self.uid = 'random-gaussian'
        self.mode = 'regression'
        self.condition = None
        self.tuning_params = {}
        self.mu = 0.5
        self.sigma = 1/6
        self.minimum = 0
        self.maximum = 1
        self.header = {
            'Author': 'A. Derington',
            'Date': '2023-06-16',
            'Name': 'random-gaussian',
            'Parameters': {
                'Mu': self.mu,
                'Sigma': self.sigma,
                'Minimum': self.minimum,
                'Maximum': self.maximum,
                'Seed': SEED
            }
        }

    def process_index(self, index: pd.Index) -> pd.Series:
        r"""Prediction created from sampling a Gaussian Distribution"""
        n_samples = len(index)
        predictions = {}
        for condition in ['arousal', 'dominance', 'valence']:
            s = truncnorm.rvs(
                (self.minimum-self.mu)/self.sigma,
                (self.maximum-self.mu)/self.sigma,
                loc=self.mu, scale=self.sigma,
                size=n_samples
            )
            y = pd.Series(data=s, index=index)
            predictions[condition] = y
        df = pd.DataFrame(data=predictions, index=index)
        return df


class RandomUniformCategorical:
    r"""Return predictions from a model with uniformly distributed categories"""

    def __init__(self) -> None:
        self.uid = 'random-categorical'
        self.mode = 'classification'
        self.condition = None
        self.tuning_params = {}
        self.header = {
            'Author': 'A. Derington',
            'Date': '2023-06-16',
            'Name': 'random-categorical',
            'Parameters': {
                'Seed': SEED
            }
        }

    def process_index(self, index: pd.Index) -> pd.Series:
        r"""Prediction created from sampling uniformly distributed categories"""
        labels = CATEGORY_LABELS[self.condition]
        n_samples = len(index)
        s = np.random.choice(a=labels, size=n_samples)
        y = pd.Series(data=s, index=index)
        y.name = self.condition
        return y.to_frame()
