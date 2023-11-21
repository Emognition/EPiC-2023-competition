#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: sequential.py
# Created Date: Friday, April 21st 2023, 8:40:35 am
# Author: Chirag Raman
#
# Copyright (c) 2023 Chirag Raman
###


import math
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class SequenceEncoder(nn.Module):

    """Encode sequence inputs using a GRU and sample output to match ground truth sampling."""

    def __init__(
            self, ninp: int, nhid: int, nout: int, ninp_per_output: int = 1,
            nlayers: int = 1, dropout: float = 0, nlinear_layers: int = 2,
            nlinear_hid: int = 64, act_type: Type[nn.Module] = nn.ReLU
        ) -> None:
        """ Initialize the model """
        super().__init__()

        # GRU backbone
        self.gru = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
        # Output MLP
        dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        fc_layers = [nn.Linear(nhid, nlinear_hid, bias=True), act_type(), dropout_layer]
        fc_layers.extend(
            [item for _ in range(1, nlinear_layers)
            for item in [nn.Linear(nlinear_hid, nlinear_hid, bias=True), act_type(), dropout_layer]]
        )
        fc_layers.append(nn.Linear(nlinear_hid, nout, bias=True))
        self.out_linear = nn.Sequential(*fc_layers)
        self.hid_linear = nn.Linear(nhid, nout)
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.stride = ninp_per_output
        self.nlayers = nlayers

    def forward(self, inputs: Tensor) -> Tensor:
        """ Forward pass to encode the sequence.

        Args:
            inputs  -- tensor (seq_len, batch_size, ninp)

        Returns encoded sequence output and last hidden state of the gru
        output  -- tensor (seq_len//self.stride, batch_size, self.nout)
        hidden  -- tensor (nlayers, batch_size, self.nout)
        """
        output, h_n = self.gru(inputs)
        # Sample from the output to match ground truth sampling
        sampled_output = output[::self.stride]
        sampled_output = self.out_linear(sampled_output)
        # Assuming uni-directional, so last dim is nhid
        h_n = self.hid_linear(h_n)

        return sampled_output#, h_n

    def init_hidden(self, batch_size):
        """ Initialise the hidden tensors """
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, batch_size, self.nhid)