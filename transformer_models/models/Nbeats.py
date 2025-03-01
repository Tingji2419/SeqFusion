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
N-BEATS Model.
"""
from typing import Tuple

import numpy as np
import torch as t
import torch.nn as nn

class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        if input_mask:
            residuals = x.flip(dims=(1,))
            input_mask = input_mask.flip(dims=(1,))
            forecast = x[:, -1:]
            for i, block in enumerate(self.blocks):
                backcast, block_forecast = block(residuals)
                residuals = (residuals - backcast) * input_mask
                forecast = forecast + block_forecast
        else:
            residuals = x.flip(dims=(1,))
            forecast = x[:, -1:]
            for i, block in enumerate(self.blocks):
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                forecast = forecast + block_forecast
        return forecast


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


def generic(input_size: int, output_size: int,
            stacks: int, layers: int, layer_size: int):
    """
    Create N-BEATS generic model.s
    """
    return NBeats(t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                               theta_size=input_size + output_size,
                                               basis_function=GenericBasis(backcast_size=input_size,
                                                                           forecast_size=output_size),
                                               layers=layers,
                                               layer_size=layer_size)
                                   for _ in range(stacks)]))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.encoder = generic(
            input_size=configs.seq_len,
            output_size=configs.pred_len,
            stacks=configs.e_layers,
            layers=configs.d_layers,
            layer_size=configs.d_model
        )

    def forecast(self, x_enc):
        b, t, c = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(b*c, t)
        # Encoder
        return self.encoder(x_enc, input_mask=None).reshape(b, c, -1).permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Encoder
        return self.forecast(x_enc)
