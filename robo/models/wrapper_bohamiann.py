import numpy as np
import torch

from pybnn.bohamiann import Bohamiann
from pybnn.multi_task_bohamiann import MultiTaskBohamiann

from robo.models.base_model import BaseModel


class WrapperBohamiann(BaseModel):

    def __init__(self, get_net=get_default_network, transformer=lambda x:x, lr=1e-2, use_double_precision=True, verbose=True):
        """
        Wrapper around pybnn Bohamiann implementation. It automatically adjusts the length by the MCMC chain,
        by performing 100 times more burnin steps than we have data points and sampling ~100 networks weights.

        Parameters
        ----------
        get_net: func
            Architecture specification

        lr: float
           The MCMC step length

        use_double_precision: Boolean
           Use float32 or float64 precision. Note: Using float64 makes the training slower.

        verbose: Boolean
           Determines whether to print pybnn output.
        """

        self.lr = lr
        self.verbose = verbose
        self.transformer = transformer
        self.bnn = Bohamiann(get_network=get_net, use_double_precision=use_double_precision)

    def train(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.bnn.train(self.transformer(X), y, lr=self.lr,
                       num_burn_in_steps=X.shape[0] * 100,
                       num_steps=X.shape[0] * 100 + 10000, verbose=self.verbose)

    def predict(self, X_test):
        return self.bnn.predict(self.transformer(X_test))


class WrapperBohamiannMultiTask(BaseModel):

    def __init__(self, n_tasks=2, transformer=lambda x:x, lr=1e-2, use_double_precision=True, verbose=False):
        """
        Wrapper around pybnn Bohamiann implementation. It automatically adjusts the length by the MCMC chain,
        by performing 100 times more burnin steps than we have data points and sampling ~100 networks weights.

        Parameters
        ----------
        get_net: func
            Architecture specification

        lr: float
           The MCMC step length

        use_double_precision: Boolean
           Use float32 or float64 precision. Note: Using float64 makes the training slower.

        verbose: Boolean
           Determines whether to print pybnn output.
        """

        self.lr = lr
        self.verbose = verbose
        self.transformer = transformer
        self.bnn = MultiTaskBohamiann(n_tasks,
                                      use_double_precision=use_double_precision)

    def train(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.bnn.train(self.transformer(X), y, lr=self.lr, mdecay=0.01,
                       num_burn_in_steps=X.shape[0] * 500,
                       num_steps=X.shape[0] * 500 + 10000, verbose=self.verbose)

    def predict(self, X_test):
        return self.bnn.predict(self.transformer(X_test))
