# -*- coding: utf-8 -*-
"""
Created on 14 Oct 2020 14:19:19
https://github.com/ruotianluo/self-critical.pytorch/tree/3.2

This file contains UpDown model

UpDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
https://arxiv.org/abs/1707.07998
However, it may not be identical to the author's architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser, _ArgumentGroup
from typing import Union, Dict
from functools import reduce
from models import register_model
from models.caption_model import CaptionModel
from data.collate import AttCollate
from utils.model_utils import repeat_tensors, pack_wrapper, filter_model_inputs
from tokenizer import Tokenizer

bad_endings = [
    'a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
    'are', 'am', 'the'
]


# noinspection PyAbstractClass,PyAttributeOutsideInit,PyMethodMayBeStatic
class AttModel(CaptionModel):
    def __init__(self, config, tokenizer: Tokenizer = None):
        super(AttModel, self).__init__()
        self.config = config
        self.input_encoding_size = config.input_encoding_size
        self.rnn_size = config.rnn_size
        # self.num_layers = config.num_layers
        self.drop_prob_lm = config.drop_prob_lm
        self.seq_length = config.max_seq_length
        self.fc_feat_size = config.fc_feat_size
        self.att_feat_size = config.att_feat_size
        self.att_hid_size = config.att_hid_size
        self.vocab_size = config.vocab_size
        self.eos_idx = config.eos_token_id
        self.bos_idx = config.bos_token_id
        self.unk_idx = config.unk_token_id
        self.pad_idx = config.pad_token_id

        self.use_bn = config.get('use_bn', 0)
        self.ss_prob = 0.0  # Schedule sampling probability

        # For remove bad ending
        if tokenizer is None:
            self.bad_endings_ix = []
        else:
            self.bad_endings_ix = [tokenizer.token_to_id(w) for w in bad_endings]
        self.make_model()

    def forward(self, input_dict: Dict, **kwargs):
        inputs = filter_model_inputs(
            input_dict=input_dict,
            mode=kwargs.get("mode", "forward"),
            required_keys=("fc_feats", "att_feats", "att_masks"),
            forward_keys=("seqs",)
        )
        return super().forward(**inputs, **kwargs)

    def make_model(self):
        self.embed = nn.Sequential(
            nn.Embedding(self.vocab_size + 1, self.input_encoding_size),  # TODO: fix vocab_size, remove + 1
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        self.fc_embed = nn.Sequential(
            nn.Linear(self.fc_feat_size, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        self.att_embed = nn.Sequential(
            *(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (
                        nn.Linear(self.att_feat_size, self.rnn_size),
                        nn.ReLU(),
                        nn.Dropout(self.drop_prob_lm)
                    ) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())
            )
        )

        self.logit_layers = self.config.get('logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [
                [nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)]
                for _ in range(self.config.logit_layers - 1)
            ]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)])
            )
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = self.logit.weight \
            if hasattr(self.logit, "weight") \
            else self.logit[0].weight
        return (
            weight.new_zeros(self.num_layers, bsz, self.rnn_size),
            weight.new_zeros(self.num_layers, bsz, self.rnn_size)
        )

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation consumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seqs, att_masks=None):
        batch_size = fc_feats.size(0)
        if seqs.ndim == 3:  # B * seq_per_img * seq_len
            seqs = seqs.reshape(-1, seqs.shape[2])
        seq_per_img = seqs.shape[0] // batch_size
        state = self.init_hidden(batch_size * seq_per_img)

        outputs = fc_feats.new_zeros(batch_size * seq_per_img, seqs.size(1), self.vocab_size + 1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(
                seq_per_img,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        for i in range(seqs.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.new(batch_size * seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seqs[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seqs[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seqs[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seqs[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _sample(self, fc_feats, att_feats, att_masks=None, opt=None):
        if opt is None:
            opt = {}
        num_random_sample = opt.get("num_random_sample", 0)
        beam_size = opt.get("beam_size", 1)
        temperature = opt.get("temperature", 1.0)
        decoding_constraint = opt.get("decoding_constraint", 0)
        batch_size = att_feats.shape[0]

        fc_feats, att_feats, p_att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        state = self.init_hidden(batch_size)

        if num_random_sample <= 0 and beam_size > 1:
            assert beam_size <= self.vocab_size + 1
            it = att_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
            seq_logprobs = att_feats.new_zeros(batch_size, beam_size, self.seq_length)
            seq = att_feats.new_full((batch_size, beam_size, self.seq_length), self.pad_idx, dtype=torch.long)

            # first step, feed bos
            logprobs, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            fc_feats, att_feats, p_att_feats, att_masks = repeat_tensors(
                beam_size,
                [fc_feats, att_feats, p_att_feats, att_masks]
            )
            self.done_beams = self.batch_beam_search(
                state, logprobs, fc_feats, att_feats, p_att_feats, att_masks, opt=opt
            )

            for k in range(batch_size):
                for b in range(beam_size):
                    res = self.done_beams[k][b]
                    seq_len = res["seq"].shape[0]
                    seq[k, b, :seq_len] = res["seq"]
                    seq_logprobs[k, b, :seq_len] = res["logps"].gather(1, res["seq"].unsqueeze(1)).squeeze(1)
                # top_seq = self.done_beams[k][0]["seq"]
                # seq_len = top_seq.shape[0]
                # seq[k, :seq_len] = top_seq  # the first beam has highest cumulative score
                # seq_logprobs[k, :seq_len] = self.done_beams[k][0]["logps"].gather(1, top_seq.unsqueeze(1)).squeeze(1)
            # return the samples and their log likelihoods
            return seq, seq_logprobs

        # Greedy search or random sample
        if num_random_sample > 0:
            assert beam_size < 1, f"Beam size must be < 1, saw {beam_size}"
            batch_size *= num_random_sample
            fc_feats, att_feats, p_att_feats, att_masks = repeat_tensors(
                n=num_random_sample, x=(fc_feats, att_feats, p_att_feats, att_masks)
            )
            # (self.num_layers, bsz, self.rnn_size)
            state = repeat_tensors(n=num_random_sample, x=state, dim=1)
            # state = tuple(_.repeat_interleave(num_random_sample, dim=1) for _ in state)
        else:
            assert beam_size == 1, f"Beam size must be 1, saw {beam_size}"

        it = att_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        seq_logprobs = att_feats.new_zeros(batch_size, self.seq_length)
        seq = att_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long)

        unfinished = it != self.eos_idx
        for t in range(self.seq_length + 1):
            logprobs, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(batch_size, self.vocab_size + 1)
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float("-inf"))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if num_random_sample > 0:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing
            else:
                # greedy search
                sample_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()

            # stop when all finished
            seq[:, t] = it * unfinished.type_as(it)
            unfinished = unfinished * (it != self.eos_idx)
            seq_logprobs[:, t] = sample_logprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        if num_random_sample > 0:
            seq = seq.view(-1, num_random_sample, self.seq_length)
            seq_logprobs = seq_logprobs.view(-1, num_random_sample, self.seq_length)
        else:
            seq = seq.view(-1, 1, self.seq_length)
            seq_logprobs = seq_logprobs.view(-1, 1, self.seq_length)
        return seq, seq_logprobs

    # def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt=None):
    #     if opt is None:
    #         opt = {}
    #     beam_size = opt.get('beam_size', 10)
    #     group_size = opt.get('group_size', 1)
    #     sample_n = opt.get('sample_n', 10)
    #     # when sample_n == beam_size then each beam is a sample.
    #     assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
    #     batch_size = fc_feats.size(0)
    #
    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
    #
    #     assert beam_size <= self.vocab_size + 1, (
    #         'lets assume this for now, otherwise this corner case causes a few headaches down the road. '
    #         'can be dealt with in future if needed'
    #     )
    #     seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
    #     seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
    #     # lets process every image independently for now, for simplicity
    #
    #     self.done_beams = [[] for _ in range(batch_size)]
    #
    #     state = self.init_hidden(batch_size)
    #
    #     # first step, feed bos
    #     it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
    #     logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
    #
    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(
    #         beam_size,
    #         [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
    #     )
    #     self.done_beams = self.batch_beam_search(
    #         state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt
    #     )
    #     for k in range(batch_size):
    #         if sample_n == beam_size:
    #             for _n in range(sample_n):
    #                 seq_len = self.done_beams[k][_n]['seq'].shape[0]
    #                 seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
    #                 seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
    #         else:
    #             seq_len = self.done_beams[k][0]['seq'].shape[0]
    #             seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
    #             seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
    #     # return the samples and their log likelihoods
    #     return seq, seqLogprobs
    #
    # def _sample(self, fc_feats, att_feats, att_masks=None, opt=None):
    #     if opt is None:
    #         opt = {}
    #     sample_method = opt.get('sample_method', 'greedy')
    #     beam_size = opt.get('beam_size', 1)
    #     temperature = opt.get('temperature', 1.0)
    #     sample_n = int(opt.get('sample_n', 1))
    #     group_size = opt.get('group_size', 1)
    #     output_logsoftmax = opt.get('output_logsoftmax', 1)
    #     decoding_constraint = opt.get('decoding_constraint', 0)
    #     block_trigrams = opt.get('block_trigrams', 0)
    #     remove_bad_endings = opt.get('remove_bad_endings', 0) and len(self.bad_endings_ix) > 0
    #     if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
    #         return self._sample_beam(fc_feats, att_feats, att_masks, opt)
    #     if group_size > 1:
    #         return self._diverse_sample(fc_feats, att_feats, att_masks, opt)
    #
    #     batch_size = fc_feats.size(0)
    #     state = self.init_hidden(batch_size * sample_n)
    #
    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
    #
    #     if sample_n > 1:
    #         p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(
    #             sample_n,
    #             [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
    #         )
    #
    #     trigrams = []  # will be a list of batch_size dictionaries
    #
    #     seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
    #     seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
    #     for t in range(self.seq_length + 1):
    #         if t == 0:  # input <bos>
    #             it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)
    #
    #         logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
    #                                                   output_logsoftmax=output_logsoftmax)
    #
    #         if decoding_constraint and t > 0:
    #             tmp = logprobs.new_zeros(logprobs.size())
    #             tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
    #             logprobs = logprobs + tmp
    #
    #         if remove_bad_endings and t > 0:
    #             tmp = logprobs.new_zeros(logprobs.size())
    #             prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
    #             # Make it impossible to generate bad_endings
    #             tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
    #             logprobs = logprobs + tmp
    #
    #         # Mess with trigrams
    #         # Copy from https://github.com/lukemelas/image-paragraph-captioning
    #         if block_trigrams and t >= 3:
    #             # Store trigram generated at last step
    #             prev_two_batch = seq[:, t - 3:t - 1]
    #             for i in range(batch_size):  # = seq.size(0)
    #                 prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
    #                 current = seq[i][t - 1]
    #                 if t == 3:  # initialize
    #                     trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
    #                 elif t > 3:
    #                     if prev_two in trigrams[i]:  # add to list
    #                         trigrams[i][prev_two].append(current)
    #                     else:  # create list
    #                         trigrams[i][prev_two] = [current]
    #             # Block used trigrams at next step
    #             prev_two_batch = seq[:, t - 2:t]
    #             mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device)  # batch_size x vocab_size
    #             for i in range(batch_size):
    #                 prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
    #                 if prev_two in trigrams[i]:
    #                     for j in trigrams[i][prev_two]:
    #                         mask[i, j] += 1
    #             # Apply mask to log probs
    #             # logprobs = logprobs - (mask * 1e9)
    #             alpha = 2.0  # = 4
    #             logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)
    #
    #         # sample the next word
    #         if t == self.seq_length:  # skip if we achieve maximum length
    #             break
    #         it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)
    #
    #         # stop when all finished
    #         if t == 0:
    #             unfinished = it != self.eos_idx
    #         else:
    #             it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
    #             logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
    #             unfinished = unfinished & (it != self.eos_idx)
    #         seq[:, t] = it
    #         seqLogprobs[:, t] = logprobs
    #         # quit loop if all sequences have finished
    #         if unfinished.sum() == 0:
    #             break
    #
    #     return seq, seqLogprobs
    #
    # def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt=None):
    #     if opt is None:
    #         opt = {}
    #     sample_method = opt.get('sample_method', 'greedy')
    #     beam_size = opt.get('beam_size', 1)
    #     temperature = opt.get('temperature', 1.0)
    #     group_size = opt.get('group_size', 1)
    #     diversity_lambda = opt.get('diversity_lambda', 0.5)
    #     decoding_constraint = opt.get('decoding_constraint', 0)
    #     block_trigrams = opt.get('block_trigrams', 0)
    #     remove_bad_endings = opt.get('remove_bad_endings', 0) and len(self.bad_endings_ix) > 0
    #
    #     batch_size = fc_feats.size(0)
    #     state = self.init_hidden(batch_size)
    #
    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
    #
    #     trigrams_table = [[] for _ in range(group_size)]  # will be a list of batch_size dictionaries
    #
    #     seq_table = [fc_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long) for _ in
    #                  range(group_size)]
    #     seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.seq_length) for _ in range(group_size)]
    #     state_table = [self.init_hidden(batch_size) for _ in range(group_size)]
    #
    #     for tt in range(self.seq_length + group_size):
    #         for divm in range(group_size):
    #             t = tt - divm
    #             seq = seq_table[divm]
    #             seqLogprobs = seqLogprobs_table[divm]
    #             trigrams = trigrams_table[divm]
    #             if t >= 0 and t <= self.seq_length - 1:
    #                 if t == 0:  # input <bos>
    #                     it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
    #                 else:
    #                     it = seq[:, t - 1]  # changed
    #
    #                 logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,
    #                                                                       p_att_masks, state_table[divm])  # changed
    #                 logprobs = F.log_softmax(logprobs / temperature, dim=-1)
    #
    #                 # Add diversity
    #                 if divm > 0:
    #                     unaug_logprobs = logprobs.clone()
    #                     for prev_choice in range(divm):
    #                         prev_decisions = seq_table[prev_choice][:, t]
    #                         logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda
    #
    #                 if decoding_constraint and t > 0:
    #                     tmp = logprobs.new_zeros(logprobs.size())
    #                     tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
    #                     logprobs = logprobs + tmp
    #
    #                 if remove_bad_endings and t > 0:
    #                     tmp = logprobs.new_zeros(logprobs.size())
    #                     prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
    #                     # Impossible to generate remove_bad_endings
    #                     tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
    #                     logprobs = logprobs + tmp
    #
    #                 # Mess with trigrams
    #                 if block_trigrams and t >= 3:
    #                     # Store trigram generated at last step
    #                     prev_two_batch = seq[:, t - 3:t - 1]
    #                     for i in range(batch_size):  # = seq.size(0)
    #                         prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
    #                         current = seq[i][t - 1]
    #                         if t == 3:  # initialize
    #                             trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
    #                         elif t > 3:
    #                             if prev_two in trigrams[i]:  # add to list
    #                                 trigrams[i][prev_two].append(current)
    #                             else:  # create list
    #                                 trigrams[i][prev_two] = [current]
    #                     # Block used trigrams at next step
    #                     prev_two_batch = seq[:, t - 2:t]
    #                     mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
    #                     for i in range(batch_size):
    #                         prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
    #                         if prev_two in trigrams[i]:
    #                             for j in trigrams[i][prev_two]:
    #                                 mask[i, j] += 1
    #                     # Apply mask to log probs
    #                     # logprobs = logprobs - (mask * 1e9)
    #                     alpha = 2.0  # = 4
    #                     logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)
    #
    #                 it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)
    #
    #                 # stop when all finished
    #                 if t == 0:
    #                     unfinished = it != self.eos_idx
    #                 else:
    #                     unfinished = seq[:, t - 1] != self.pad_idx & seq[:, t - 1] != self.eos_idx
    #                     it[~unfinished] = self.pad_idx
    #                     unfinished = unfinished & (it != self.eos_idx)  # changed
    #                 seq[:, t] = it
    #                 seqLogprobs[:, t] = sampleLogprobs.view(-1)
    #
    #     return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table,
    #                                                                                        1).reshape(
    #         batch_size * group_size, -1)


# noinspection PyAbstractClass
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rnn_size = config.rnn_size
        self.att_hid_size = config.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


# noinspection PyAbstractClass
class UpDownCore(nn.Module):
    def __init__(self, config, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.config = config
        self.drop_prob_lm = config.drop_prob_lm

        self.att_lstm = nn.LSTMCell(
            config.input_encoding_size + config.rnn_size * 2, config.rnn_size
        )  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(config.rnn_size * 2, config.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(config)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


# noinspection PyAbstractClass,PyAttributeOutsideInit
@register_model("up_down_lstm")
class UpDownModel(AttModel):
    COLLATE_FN = AttCollate

    def __init__(self, config, tokenizer: Tokenizer = None):
        self.num_layers = 2
        super().__init__(config, tokenizer)
        self.core = UpDownCore(self.config)

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        # CaptionModel.add_argparse_args(parser)
        # CaptionModel args
        parser.add_argument(
            "--max_seq_length", type=int, default=16,
            help="int: Maximum sequence length excluding <bos> and <eos>.",
        )
        parser.add_argument(
            "--rnn_size", type=int, default=1000,
            help="int: Size of the RNN (number of units)."
        )
        # parser.add_argument(
        #     "--num_layers", type=int, default=6,
        #     help="int: Number of RNN layers in the model"
        # )
        parser.add_argument(
            "--input_encoding_size", type=int, default=1000,
            help="int: The encoding size of each token in the vocabulary, and the image."
        )
        parser.add_argument(
            "--att_feat_size", type=int, default=2048,
            help="int: 2048 for resnet, 512 for vgg"
        )
        parser.add_argument(
            "--drop_prob_lm", type=float, default=0.5,
            help="float: Strength of dropout in the Language Model RNN"
        )
        # AttModel args
        parser.add_argument(
            "--att_hid_size", type=int, default=512,
            help="int: The hidden size of the attention MLP for show_attend_tell; 0 if not using hidden layer"
        )
        parser.add_argument(
            "--fc_feat_size", type=int, default=2048,
            help="int: 2048 for resnet, 4096 for vgg"
        )
        parser.add_argument(
            "--logit_layers", type=int, default=1,
            help="int: Number of layers in the RNN"
        )
        # fmt: on
        # return parser
