import pdb
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import math
import enum
import warnings
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import TabularIndivBase, IndivTabDataset
from ..losses import XentLoss, BinaryXentLoss, MultilabelBinaryXentLoss, MSELoss

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'

def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].
    Examples:
        .. testcode::
            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)
    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].
    Examples:
        .. testcode::
            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)
    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)

class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)

class CategoricalFeatureTokenizer(nn.Module):
    """Transforms categorical features to tokens (embeddings).
    See `FeatureTokenizer` for the illustration.
    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).
    Examples:
        .. testcode::
            # the input must contain integers. For example, if the first feature can
            # take 3 distinct values, then its cardinality is 3 and the first column
            # must contain values from the range `[0, 1, 2]`.
            cardinalities = [3, 10]
            x = torch.tensor([
                [0, 5],
                [1, 7],
                [0, 2],
                [2, 4]
            ])
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    category_offsets: Tensor

    def __init__(
        self,
        cardinalities: List[int],
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                :code:`cardinalities=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: the size of one token.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class NumericalFeatureTokenizer(nn.Module):
    """Transforms continuous features to tokens (embeddings).
    See `FeatureTokenizer` for the illustration.
    For one feature, the transformation consists of two steps:
    * the feature is multiplied by a trainable vector
    * another trainable vector is added
    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        """
        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there
                are no numerical features.
            cat_cardinalities: the number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, 'n_num_features must be non-negative'
        assert (
            n_num_features or cat_cardinalities
        ), 'at least one of n_num_features or cat_cardinalities must be positive/non-empty'
        self.initialization = 'uniform'
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=n_num_features,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if n_num_features
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(
            x.n_tokens
            for x in [self.num_tokenizer, self.cat_tokenizer]
            if x is not None
        )

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return (
            self.cat_tokenizer.d_token  # type: ignore
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Perform the forward pass.
        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0`
                was passed to the constructor.
            x_cat: categorical features (see `CategoricalFeatureTokenizer.forward` for
                details). Must be presented if non-empty :code:`cat_cardinalities` was
                passed to the constructor.
        Returns:
            tokens
        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        assert (
            x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.
    To learn about the [CLS]-based inference, see [devlin2018bert].
    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.
    Examples:
        .. testcode::
            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()
    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) -> None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.
        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.
        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.
        Args:
            leading_dimensions: the additional new dimensions
        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)

class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' attention.
    To learn more about Multihead Attention, see [devlin2018bert]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [wang2020linformer] to speed up the module when the number of tokens is large.
    Examples:
        .. testcode::
            n_objects, n_tokens, d_token = 2, 3, 12
            n_heads = 6
            a = torch.randn(n_objects, n_tokens, d_token)
            b = torch.randn(n_objects, n_tokens * 2, d_token)
            module = MultiheadAttention(
                d_token=d_token, n_heads=n_heads, dropout=0.2, bias=True, initialization='kaiming'
            )
            # self-attention
            x, attention_stats = module(a, a, None, None)
            assert x.shape == a.shape
            assert attention_stats['attention_probs'].shape == (n_objects * n_heads, n_tokens, n_tokens)
            assert attention_stats['attention_logits'].shape == (n_objects * n_heads, n_tokens, n_tokens)
            # cross-attention
            assert module(a, b, None, None)
            # Linformer self-attention with the 'headwise' sharing policy
            k_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            v_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            assert module(a, a, k_compression, v_compression)
            # Linformer self-attention with the 'key-value' sharing policy
            kv_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            assert module(a, a, kv_compression, kv_compression)
    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            d_token: the token size. Must be a multiple of :code:`n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
            bias: if `True`, then input (and output, if presented) layers also have bias.
                `True` is a reasonable default choice.
            initialization: initialization for input projection layers. Must be one of
                :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        Raises:
            AssertionError: if requirements for the inputs are not met.
        """
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == 'xavier' and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform the forward pass.
        Args:
            x_q: query tokens
            x_kv: key-value tokens
            key_compression: Linformer-style compression for keys
            value_compression: Linformer-style compression for values
        Returns:
            (tokens, attention_stats)
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), 'If key_compression is (not) None, then value_compression must (not) be None'
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            'attention_logits': attention_logits,
            'attention_probs': attention_probs,
        }

class Transformer(nn.Module):
    """Transformer with extra features.
    This module is the backbone of `FTTransformer`."""

    WARNINGS = {'first_prenormalization': True, 'prenormalization': True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: Union[None, List[int], slice],
        n_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                'last_layer_query_idx must be None, list[int] or slice. '
                f'Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?'
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), 'If `prenormalization` is False, then `first_prenormalization` must be False'
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            'If any of the following arguments is (not) None, then all of them must (not) be None: '
            'n_tokens, kv_compression_ratio, kv_compression_sharing'
        )
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn(
                    'prenormalization is set to False. Are you sure about this? '
                    'The training can become less stable. '
                    'You can turn off this warning by tweaking the '
                    'rtdl.Transformer.WARNINGS dictionary.',
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        if (
            prenormalization
            and first_prenormalization
            and self.WARNINGS['first_prenormalization']
        ):
            warnings.warn(
                'first_prenormalization is set to True. Are you sure about this? '
                'For example, the vanilla FTTransformer with '
                'first_prenormalization=True performs SIGNIFICANTLY worse. '
                'You can turn off this warning by tweaking the '
                'rtdl.Transformer.WARNINGS dictionary.',
                UserWarning,
            )
            time.sleep(3)

        def make_kv_compression():
            assert (
                n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    attention_normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert (
                        kv_compression_sharing == 'key-value'
                    ), _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.head = Transformer.Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
            activation=head_activation,  # type: ignore
            normalization=head_normalization if prenormalization else 'Identity',
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            )
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)

        x = self.head(x)
        return x


class BuildModel(nn.Module):
    def __init__(self,
        n_num_features,
        cat_cardinalities,
        output_dim,
        hidden_dim=128,
        num_layer=2,
        attention_dropout=0,
        attention_n_heads=8,
        attention_initialization='kaiming',
        attention_normalization='LayerNorm',
        ffn_dim=256,
        ffn_dropout=0,
        ffn_activation='ReGLU',
        ffn_normalization='LayerNorm',
        residual_dropout=0,
        prenormalization=True,
        first_prenormalization=False,
        last_layer_query_idx=None,
        n_tokens=None,
        kv_compression_ratio=None,
        kv_compression_sharing=None,
        head_activation='ReLU',
        head_normalization='LayerNorm',
        **kwargs,
        ) -> None:
        super().__init__()
        
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=hidden_dim,
        )
        self.cls_token = CLSToken(
            self.feature_tokenizer.d_token, self.feature_tokenizer.initialization
        )

        self.transformer = Transformer(
            d_token=hidden_dim,
            n_blocks=num_layer,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization=attention_initialization,
            attention_normalization=attention_normalization,
            ffn_d_hidden=ffn_dim,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=ffn_normalization,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            first_prenormalization=first_prenormalization,
            last_layer_query_idx=last_layer_query_idx,
            n_tokens=n_tokens,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=output_dim,
            )
    
    def forward(self, x_num, x_cat):
        # ensure correct data type for embedding
        x_cat = x_cat.long() 
        x_num = x_num.float()
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.cls_token(x)
        x = self.transformer(x)
        return x

class FTTransformer(TabularIndivBase):
    '''
    Implement ft-transformer model for tabular individual outcome
    prediction in clinical trials [1]_. 

    Parameters
    ----------
    num_feat: list[str]
        the list of numerical feature names.

    cat_feat: list[str]
        the list of categorical feature names.

    cat_cardinalities: list[int]
        A list of categorical features' cardinalities.

    output_dim: int
        Dimension of the outputs. When doing classification, it equals to number of classes.

    mode: str
        The task's objectives, in `binary`, `multiclass`, `multilabel`, or `regression`

    hidden_dim: int
        Hidden dimensions of neural networks. Must be a multiple of :code:`n_heads=8`.

    num_layer: int
        Number of hidden layers.

    attention_dropout: float
        the dropout for attention blocks.
        Usually, positive values work better (even when the number of features is low).

    ffn_dim: int
        the *input* size for the *second* linear layer in `Transformer.FFN`.
        Note that it can be different from the output size of the first linear
        layer, since activations such as ReGLU or GEGLU change the size of input.
        For example, if :code:`ffn_d_hidden=10` and the activation is ReGLU (which
        is always true for the baseline and default configurations), then the
        output size of the first linear layer will be set to :code:`20`.

    ffn_dropout: float
        the dropout rate after the first linear layer in `Transformer.FFN`.

    residual_dropout: float
        the dropout rate for the output of each residual branch of all Transformer blocks.

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    weight_decay: float
        Regularization strength for l2 norm; must be a positive float.
        Smaller values specify weaker regularization.
    
    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        Target device to train the model, as `cuda:0` or `cpu`.

    experiment_id: str, optional (default='test')
        The name of current experiment. Decide the saved model checkpoint name.
    
    Notes
    -----
    .. [1] Gorishniy, Y., et al. (2021). Revisiting deep learning models for tabular data. NeurIPS'21.
    '''
    def __init__(self,
        num_feat,
        cat_feat,
        cat_cardinalities,
        output_dim,
        mode,
        hidden_dim=128,
        num_layer=2,
        attention_dropout=0,
        ffn_dim=256,
        ffn_dropout=0,
        residual_dropout=0,
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        epochs=10,
        num_worker=0,
        device='cuda:0',
        experiment_id='test'):
        super().__init__(experiment_id)

        mode = mode.lower()
        assert mode in ['binary', 'multiclass', 'regression', 'multilabel']

        self.config = {
            'num_feat':num_feat,
            'cat_feat':cat_feat,
            'n_num_features':len(num_feat),
            'cat_cardinalities':cat_cardinalities,
            'output_dim':output_dim,
            'hidden_dim':hidden_dim,
            'num_layer':num_layer,
            'attention_dropout':attention_dropout,
            'ffn_dim':ffn_dim,
            'ffn_dropout':ffn_dropout,
            'residual_dropout':residual_dropout,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'weight_decay':weight_decay,
            'epochs':epochs,
            'num_worker':num_worker,
            'experiment_id':experiment_id,
            'model_name': 'MLP',
            'device':device,
            'mode':mode,
        }
        self._save_config(self.config)
        self.device = device

    def fit(self, train_data, valid_data=None):
        '''Train FT-Transformer model to predict patient outcome
        with tabular input data.

        Parameters
        ----------
        train_data: dict
            {
            'x': TabularPatientBase or pd.DataFrame,
            'y': pd.Series or np.ndarray
            }
            
            - 'x' contain all patient features; 
            - 'y' contain label for each row.

        valid_data: dict
            Same as `train_data`.
            Validation data during the training for early stopping.
        '''

        self._input_data_check(train_data)
        self._build_model()
        
        x_feat, y = self._parse_input_data(train_data)
        train_data={'x':x_feat, 'y':y}

        if valid_data is not None:
            x_feat_va, y_va = self._parse_input_data(valid_data)
            valid_data = {'x':x_feat_va, 'y':y_va}
                
        self._fit_model(train_data=train_data, valid_data=valid_data)
    
    def predict(self, test_data):
        '''
        Make prediction probability based on the learned model.

        Parameters
        ----------
        test_data: Dict or TabularPatientBase or pd.DataFrame or torch.Tensor
            {'x': TabularPatientBase or pd.DataFrame or torch.Tensor}
            
            'x' contain all patient features.

        Returns
        -------
        ypred: np.ndarray or torch.Tensor
            Prediction probability for each patient.
            
            - For binary classification, return shape (n, );
            - For multiclass classification, return shape (n, n_class).

        '''

        if isinstance(test_data, torch.Tensor):
            return self.model(test_data)
        
        x_feat, y = self._parse_input_data(test_data)
        test_data = {'x':x_feat, 'y': y}
        dataset = self._build_dataset(test_data)
        dataloader = DataLoader(dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config['num_worker'],
            pin_memory=True,
            )
        ypred = self._predict_on_dataloader(dataloader)

        return ypred

    def save_model(self, output_dir=None):
        '''
        Save the learned ft-transformer model to the disk.

        Parameters
        ----------
        output_dir: str or None
            The dir to save the learned model.
            If set None, will save model to `self.checkout_dir`.
        '''
        if output_dir is not None:
            make_dir_if_not_exist(output_dir)
        else:
            output_dir = self.checkout_dir

        self._save_config(self.config, output_dir=output_dir)
        self._save_checkpoint({'model':self.model}, output_dir=output_dir)

    def load_model(self, checkpoint):
        '''
        Load the model from the given checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.

            - If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            - If a filepath, will load from this file.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        state_dict = torch.load(checkpoint_filename)
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)
        self.model = state_dict['model']

    def _build_model(self):
        self.model = BuildModel(
            **self.config,
            )
        self.model.to(self.config['device'])

    def _build_dataset(self, inputs):
        df = inputs['x']
        
        if 'y' in inputs: label = inputs['y']
        else: label = None

        return FTIndivTabDataset(
                df, 
                num_cols=self.config['num_feat'], 
                cat_cols=self.config['cat_feat'] , 
                label=label,
            )

    def _parse_input_data(self, inputs):
        if isinstance(inputs, dict):
            if isinstance(inputs['x'], TabularPatientBase):
                dataset = inputs['x']
                x_feat = dataset.df
                y = inputs['y']
            else:
                x_feat = inputs['x']
                y = inputs['y']                
            
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
        
        if isinstance(inputs, pd.DataFrame) or isinstance(inputs, torch.Tensor):
            x_feat, y = inputs, None
        
        if isinstance(inputs, TabularPatientBase):
            x_feat, y = inputs.df, None

        return x_feat, y

    def _build_loss_model(self):
        mode = self.config['mode']
        if mode == 'multiclass':
            return FTXentLoss(self.model)

        elif mode == 'binary':
            return FTBinaryXentLoss(self.model)

        elif mode == 'regression':
            return FTMSELoss(self.model)

        elif mode == 'multilabe':
            return FTMultilabelBinaryXentLoss(self.model)

        else:
            raise ValueError(f'Do not recognize mode `{mode}`, please correct.')

    def _predict_on_dataloader(self, dataloader):
        pred_list, label_list = [], []
        for batch in dataloader:
            x_cat, x_num = batch['x_cat'], batch['x_num']
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            if 'y' in batch: label_list.append(batch.pop('y'))
            pred = self.model(x_num, x_cat)
            pred_list.append(pred)
        pred = torch.cat(pred_list)
        label = torch.cat(label_list) if len(label_list) > 0 else None
        return {'pred':pred, 'label':label}


class FTIndivTabDataset(IndivTabDataset):
    def __init__(self, df, cat_cols, num_cols, label=None):
        self.df = df
        self.label = label
        self.cat_cols = cat_cols
        self.num_cols = num_cols
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.label is not None: 
            return {
                'x_cat': self.df.iloc[idx][self.cat_cols].values, 
                'x_num': self.df.iloc[idx][self.num_cols].values,
                'y': self.label[idx]
                }
        else: 
            return {
                'x_cat': self.df.iloc[idx][self.cat_cols].values, 
                'x_num': self.df.iloc[idx][self.num_cols].values
                }

'''
Customized loss models for FT-Transformer.
'''

class FTXentLoss(XentLoss):
    '''
    The basic xent loss
    for multi-class classification.
    '''
    def forward(self, inputs):
        logits = self.model(inputs['x_num'], inputs['x_cat'])
        loss = self.xent_loss(logits, inputs['y'])
        return {'loss_value':loss}

class FTBinaryXentLoss(BinaryXentLoss):
    '''
    The basic binary xent loss
    for binary classification.
    '''
    def forward(self, inputs):
        logits = self.model(inputs['x_num'], inputs['x_cat'])
        y = inputs['y']
        if y.shape != logits.shape:
            logits = logits.reshape(y.shape)
        loss = self.xent_loss(logits, y.float())
        return {'loss_value':loss}

class FTMultilabelBinaryXentLoss(MultilabelBinaryXentLoss):
    '''
    The basic binary xent loss
    for multilabel classification.
    '''
    def forward(self, inputs):
        logits = self.model(inputs['x_num'], inputs['x_cat'])
        loss = self.xent_loss(logits, inputs['y'].float())
        return {'loss_value':loss}


class FTMSELoss(MSELoss):
    def forward(self, inputs):
        logits = self.model(inputs['x_num'], inputs['x_cat'])
        loss = self.loss(logits, inputs['y'])
        return {'loss_value':loss}