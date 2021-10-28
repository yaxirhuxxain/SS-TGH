# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

from typing import Callable

import torch
from torch import nn


class Highway(torch.nn.Module):
    # Modified version of Highway networks
    # Highway networks (https://arxiv.org/abs/1505.00387)
    def __init__(
            self,
            input_dim: int,
            num_layers: int = 1,
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2, bias=False) for _ in range(num_layers)]
        )
        self._activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.tanh(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class MishFF(nn.Module):
    def __init__(self, d_model=300, d_ff=1200, dropout_rate=0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.mish(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class ReluFF(nn.Module):
    def __init__(self, d_model=300, d_ff=1200, dropout_rate=0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
