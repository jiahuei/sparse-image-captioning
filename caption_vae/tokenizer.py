# -*- coding: utf-8 -*-
"""
Created on 06 May 2020 14:03:56
@author: jiahuei

Copyright (c) Facebook, Inc. and its affiliates, under the BSD License
Copyright 2018 Google Inc., licensed under the Apache License, Version 2.0

Google SentencePiece:
    https://github.com/google/sentencepiece/blob/v0.1.86/python/sentencepiece_python_module_example.ipynb
    https://github.com/google/sentencepiece/blob/v0.1.86/python/README.md

TorchText:
    https://github.com/pytorch/text/blob/0.6.0/torchtext/data/utils.py#L74

Facebook Research:
    https://github.com/facebookresearch/pytext/blob/v0.3.2/pytext/data/tokenizers/tokenizer.py
"""

import os
import logging
import shutil
import stanza
from argparse import ArgumentParser, _ArgumentGroup
from abc import ABC, abstractmethod
from typing import Type, Optional, Union, List, NamedTuple
from torch import Tensor
from numpy import ndarray
from utils.config import Config
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

logger = logging.getLogger(__name__)
TOKENIZER_REGISTRY = {}


def register_tokenizer(name):
    """
    New tokenizers can be added with the :func:`register_tokenizer` function decorator.

    For example::

        @register_tokenizer('CharBPE')
        class CharBPEPreTrainedTokenizer:
            (...)

    Args:
        name (str): the name of the tokenizer
    """

    def register_tokenizer_cls(cls):
        if name in TOKENIZER_REGISTRY:
            raise ValueError(f"Cannot register duplicate tokenizer: `{name}`.")
        if not issubclass(cls, Tokenizer):
            raise ValueError(f"Tokenizer ({name}: {cls.__name__}) must extend `Tokenizer`.")
        TOKENIZER_REGISTRY[name.lower()] = cls
        return cls

    return register_tokenizer_cls


def get_tokenizer(name: str):
    name = name.lower()
    try:
        return TOKENIZER_REGISTRY[name]
    except KeyError:
        _list = "\n".join(TOKENIZER_REGISTRY.keys())
        error_mssg = f"Tokenizer specified `{name}` is invalid. Available options are: \n{_list}"
        raise ValueError(error_mssg)


class PosTagger:
    UPOS = (
        "ADJ", "ADP", "ADV", "AUX",
        "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X"
    )

    def __init__(self, pretokenized: bool = True):
        super().__init__()
        try:
            self.cache_dir = os.environ["STANZA_CACHE_DIR"]
        except KeyError:
            self.cache_dir = "/tmp/stanza_resources"
        pipeline_kwargs = dict(
            # English does not require Multi Word Expansion
            lang="en",
            dir=self.cache_dir,
            processors="tokenize,pos",
            tokenize_pretokenized=pretokenized,
            pos_batch_size=50000,
            # logging_level="DEBUG",
        )
        # noinspection PyBroadException
        try:
            self.tagger = stanza.Pipeline(**pipeline_kwargs)
        except Exception:
            stanza.download("en", dir=self.cache_dir)
            self.tagger = stanza.Pipeline(**pipeline_kwargs)

    def __call__(self, sentence: str):
        assert isinstance(sentence, str), f"Only `str` is accepted as input to {self.__class__.__name__}"
        return [[f"<{word.upos}>" for word in sents.words] for sents in self.tagger(sentence).sentences]


class Token(NamedTuple):
    value: str
    start: int
    end: int


class Tokenizer(ABC):
    processor = None
    special_token_attributes = (
        "user_defined_symbols",
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "mask_token",
        # 'sep_token',
        # 'cls_token',
        # 'user_defined_symbols_ids',
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
        "pad_token_id",
        "mask_token_id",
        # 'sep_token_id',
        # 'cls_token_id',
    )
    # control_symbols = ("<mask>",) + tuple(f"<{_}>" for _ in PosTagger.UPOS)
    control_symbols = ()

    @property
    def vocab_size(self):
        """ Vocabulary size, including special tokens."""
        # noinspection PyTypeChecker
        return len(self)

    @staticmethod
    def process_tokens(input_str, tokens, token_strip_fn=None):
        """
        Calculate start and end indices of each piece.
        This roughly doubles the time taken for tokenization from 896,357.8 token/sec to 390,008.4 token/sec.
        """
        new_tokens = []
        end = 0
        for piece in tokens:
            original_piece = piece if token_strip_fn is None else token_strip_fn(piece)
            start = input_str.find(original_piece, end)
            end = start + len(original_piece)
            new_tokens.append(Token(piece, start, end))
        return new_tokens

    @abstractmethod
    def encode(
            self,
            input_str: str,
            add_bos_eos: bool = True,
            max_seq_length: int = 16,
            sampling=False,
    ) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def encode_tokenized(
            self,
            input_list: List[str],
            add_bos_eos: bool = True,
            max_seq_length: int = 16,
    ) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, input_ids: List[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, input_str: str) -> List[Token]:
        raise NotImplementedError

    @abstractmethod
    def token_to_id(self, token):
        raise NotImplementedError

    @abstractmethod
    def id_to_token(self, token_id):
        raise NotImplementedError

    @property
    @abstractmethod
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        raise NotImplementedError

    @property
    @abstractmethod
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary."""
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def sep_token_id(self):
    #     """ Id of the separation token in the vocabulary.
    #     E.g. separate context and query in an input sequence."""
    #     raise NotImplementedError

    # @property
    # @abstractmethod
    # def cls_token_id(self):
    #     """ Id of the classification token in the vocabulary.
    #     E.g. separate context and query in an input sequence."""
    #     raise NotImplementedError

    @property
    def bos_token(self):
        """ Beginning of sentence token (string)."""
        return self.id_to_token(self.bos_token_id)

    @property
    def eos_token(self):
        """ End of sentence token (string)."""
        return self.id_to_token(self.eos_token_id)

    @property
    def unk_token(self):
        """ Unknown token (string)."""
        return self.id_to_token(self.unk_token_id)

    @property
    def pad_token(self):
        """ Padding token (string)."""
        return self.id_to_token(self.pad_token_id)

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a model with masked-language modeling."""
        return self.id_to_token(self.mask_token_id)

    # @property
    # def sep_token(self):
    #     """ Separation token (string). E.g. separate context and query in an input sequence."""
    #     return self.id_to_token(self.sep_token_id)

    # @property
    # def cls_token(self):
    #     """ Classification token (string). E.g. to extract a summary of an input sequence
    #     leveraging self-attention along the full depth of the model."""
    #     return self.id_to_token(self.cls_token_id)


# noinspection PyAttributeOutsideInit
@register_tokenizer("unigram")
class SentencePieceUnigramTokenizer(Tokenizer):
    """
    https://github.com/google/sentencepiece/blob/v0.1.86/python/sentencepiece_python_module_example.ipynb

    TrainerSpec {
      input: train_captions.txt
      input_format:
      model_prefix: m
      model_type: UNIGRAM
      vocab_size: 2000
      self_test_sample_size: 0
      character_coverage: 0.9995
      input_sentence_size: 0
      shuffle_input_sentence: 1
      seed_sentencepiece_size: 1000000
      shrinking_factor: 0.75
      max_sentence_length: 4192
      num_threads: 16
      num_sub_iterations: 2
      max_sentencepiece_length: 16
      split_by_unicode_script: 1
      split_by_number: 1
      split_by_whitespace: 1
      treat_whitespace_as_suffix: 0
      hard_vocab_limit: 1
      use_all_vocab: 0
      unk_id: 0
      bos_id: 1
      eos_id: 2
      pad_id: -1
      unk_piece: <unk>
      bos_piece: <s>
      eos_piece: </s>
      pad_piece: <pad>
      unk_surface:  Γüç
    }
    NormalizerSpec {
      name: nmt_nfkc
      add_dummy_prefix: 1
      remove_extra_whitespaces: 1
      escape_whitespaces: 1
      normalization_rule_tsv:
    }

    def set_vocabulary(self, valid_vocab):
        return _sentencepiece.SentencePieceProcessor_set_vocabulary(self, valid_vocab)

    def reset_vocabulary(self):
        return _sentencepiece.SentencePieceProcessor_reset_vocabulary(self)

    def load_vocabulary(self, filename, threshold):
        return _sentencepiece.SentencePieceProcessor_load_vocabulary(self, filename, threshold)
    """

    MODEL_TYPE = "unigram"

    def __init__(self, config):
        self.config = config
        self.tokenizer_dir = os.path.join(self.config.log_dir, "tokenizer")
        self.sp_model_path = os.path.join(self.tokenizer_dir, f"{self.MODEL_TYPE}.model")
        # Maybe reload tokenizer from another training checkpoint dir
        if not config.start_from:
            start_from = ""
        elif os.path.isfile(config.start_from):
            start_from = os.path.dirname(config.start_from)
        elif os.path.isdir(config.start_from):
            start_from = config.start_from
        else:
            raise ValueError(
                f"{self.__class__.__name__}: `config.start_from` must be either dir or file. "
                f"Got `{config.start_from}`"
            )
        src_model_path = os.path.join(start_from, "tokenizer", f"{self.MODEL_TYPE}.model")
        if not os.path.isfile(self.sp_model_path) and os.path.isfile(src_model_path):
            # shutil.copytree(os.path.dirname(src_model_path), self.tokenizer_dir)
            shutil.copy2(src_model_path, self.sp_model_path)
            shutil.copy2(
                src_model_path.replace(".model", ".vocab"),
                self.sp_model_path.replace(".model", ".vocab")
            )
            train_file = os.path.join(self.tokenizer_dir, "train_captions.txt")
            if os.path.isfile(train_file):
                os.remove(train_file)
            self.config.tokenizer_train_files = None
        # Train the tokenizer if model file is not found
        if not os.path.isfile(self.sp_model_path):
            if not isinstance(self.config.tokenizer_train_files, str):
                error_mssg = (
                    "`config.tokenizer_train_files` must be provided in absence of `sp_model_path`."
                )
                raise ValueError(error_mssg)
            sp_model_path = self.train()
            logger.info(f"{self.__class__.__name__}: Tokenizer model saved to `{sp_model_path}`.")
            assert self.sp_model_path == sp_model_path
        self._load_processor()
        # Copy tokenizer attributes over to Config
        for attr in self.special_token_attributes:
            self._update_config(attr, getattr(self, attr, None))
        self._update_config("vocab_size", len(self))
        self._update_config("num_control_symbols", len(self.control_symbols))
        self._update_config("num_special_symbols", len(self.control_symbols) + 4)
        logger.info(f"{self.__class__.__name__}: Init complete.")

    def _update_config(self, key, value):
        if key in vars(self.config):
            return
        setattr(self.config, key, value)

    def encode(
            self,
            input_str: str,
            add_bos_eos: bool = True,
            max_seq_length: int = 16,
            sampling=False,
    ) -> List[int]:
        if sampling:
            ids = self.processor.encode(
                input_str, add_bos=add_bos_eos, add_eos=add_bos_eos, out_type=int,
                enable_sampling=True, alpha=0.1, nbest_size=-1
            )
        else:
            ids = self.processor.encode(
                input_str, add_bos=add_bos_eos, add_eos=add_bos_eos, out_type=int
            )
        if max_seq_length > 0:
            ids = ids[:max_seq_length]
        # if add_bos_eos:
        #     ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def encode_tokenized(
            self,
            input_list: List[str],
            add_bos_eos: bool = True,
            max_seq_length: int = 16,
    ) -> List[int]:
        assert isinstance(input_list, list)
        ids = self.processor.piece_to_id(input_list)
        if add_bos_eos:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if max_seq_length > 0:
            ids = ids[:max_seq_length]
        return ids

    def decode(self, input_ids: List[int]) -> str:
        if isinstance(input_ids, (Tensor, ndarray)):
            if len(input_ids.shape) == 1:
                input_ids = list(map(int, input_ids))
            elif len(input_ids.shape) == 0:
                input_ids = [int(input_ids)]
            else:
                error_mssg = (
                    f"`input_tensor` can be either 1D or 0D, saw `{len(input_ids.shape)}`D instead."
                )
                raise ValueError(error_mssg)
        # TODO: remove after vocab_size is fixed
        input_ids = [_ if _ < len(self) else self.config.unk_token_id for _ in input_ids]
        sent = self.processor.decode_ids(input_ids).replace("<unk>", " <unk>")
        if sent.startswith(" "):
            sent = sent[1:]
        return sent

    def tokenize(self, input_str: str) -> List[Token]:
        pieces = self.processor.encode_as_pieces(input_str)
        return self.process_tokens(" " + input_str, pieces, lambda x: x.replace("\u2581", " "))

    def train(self):
        logger.info(f"{self.__class__.__name__}: Training on `{self.config.tokenizer_train_files}`.")
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        model_prefix = os.path.join(self.tokenizer_dir, self.MODEL_TYPE)
        # if user_defined_symbols is not None:
        #     assert isinstance(
        #         user_defined_symbols, str
        #     ), f"`user_defined_symbols` must be a `str`, saw `{type(user_defined_symbols)}` instead."
        #     assert not user_defined_symbols.startswith(
        #         ","
        #     ), f"`user_defined_symbols` must not start with a comma `,`."

        log_level = 2
        if isinstance(self.config.logging_level, int):
            log_level = int((self.config.logging_level - 10) / 10)
        SentencePieceTrainer.train(
            f"--input={self.config.tokenizer_train_files} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={self.config.vocab_size} "
            f"--hard_vocab_limit=false "  # Allow final vocab size to be smaller if training set is too small
            f"--model_type={self.MODEL_TYPE} "
            f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
            f"--pad_piece=<pad> --unk_piece=<unk> --bos_piece=<bos> --eos_piece=<eos> "
            f"--unk_surface=<unk> "
            # f"--control_symbols={','.join(self.control_symbols)} "
            # f"--user_defined_symbols={user_defined_symbols} "
            # f"--minloglevel={log_level} "
        )
        return f"{model_prefix}.model"

    def token_to_id(self, token):
        return self.processor.piece_to_id(token)

    def id_to_token(self, token_id):
        if token_id >= len(self):
            error_mssg = f"`token_id` ({token_id}) exceeded the vocab size ({len(self)}). Max ID is `{len(self) - 1}`."
            raise ValueError(error_mssg)
        return self.processor.id_to_piece(token_id)

    def _load_processor(self):
        self.processor = SentencePieceProcessor()
        self.processor.load(self.sp_model_path)

    def __len__(self):
        return len(self.processor)

    def __getstate__(self):
        # https://github.com/facebookresearch/pytext/blob/v0.3.2/pytext/data/tokenizers/tokenizer.py#L264
        state = dict(vars(self))
        state.pop("processor")
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        self._load_processor()

    def __getattr__(self, name):
        try:
            return getattr(self.processor, name)
        except AttributeError as e:
            raise e

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self.processor.bos_id()

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self.processor.eos_id()

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary."""
        return self.processor.unk_id()

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary."""
        return self.processor.pad_id()

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary."""
        return self.token_to_id("<mask>")

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        parser.add_argument(
            "--tokenizer_train_files",
            type=str,
            default=None,
            help="comma-separated str: Paths to the tokenizer training text files.",
        )
        parser.add_argument(
            "--vocab_size",
            type=int,
            default=10001,
            help="int: Maximum vocabulary size.",
        )
        # fmt: on
        # return parser


@register_tokenizer("bpe")
class SentencePieceBPETokenizer(SentencePieceUnigramTokenizer):
    """
    Character tokenizer implemented using Sentence Piece.
    """

    MODEL_TYPE = "bpe"


@register_tokenizer("character")
class CharacterTokenizer(SentencePieceUnigramTokenizer):
    """
    Character tokenizer implemented using Sentence Piece.
    """

    MODEL_TYPE = "character"


@register_tokenizer("word")
class WordTokenizer(SentencePieceUnigramTokenizer):
    """
    Word tokenizer implemented using Sentence Piece.
    """

    MODEL_TYPE = "word"
