# -*- coding: utf-8 -*-
"""
Created on 14 Oct 2020 14:32:20
"""
# import torch
# import torch.nn as nn
#
#
# class LSTMCore(nn.Module):
#     def __init__(self, opt):
#         super(LSTMCore, self).__init__()
#         self.input_encoding_size = opt.input_encoding_size
#         self.rnn_size = opt.rnn_size
#         self.drop_prob_lm = opt.drop_prob_lm
#
#         # Build a LSTM
#         self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
#         self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
#         self.dropout = nn.Dropout(self.drop_prob_lm)
#
#     def forward(self, xt, state):
#         all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
#         sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
#         sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
#         in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
#         forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
#         out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
#
#         in_transform = torch.max(
#             all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
#             all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size)
#         )
#         next_c = forget_gate * state[1][-1] + in_gate * in_transform
#         next_h = out_gate * torch.tanh(next_c)
#
#         output = self.dropout(next_h)
#         state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
#         return output, state
