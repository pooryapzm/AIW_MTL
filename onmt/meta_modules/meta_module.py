import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
import warnings
from collections import OrderedDict
from torch._six import container_abcs
from itertools import islice
import operator

_VF = torch._C._VariableFunctions


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    y=Variable(x, requires_grad=requires_grad)
    del x
    return y

# TODO: add recurse feature
class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            #print(name)
            yield param

    def parameters(self):
        for name, param in self.named_params(self):
            #print(name)
            yield param
    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                #only for debugging
                # if "embedding" in name:
                #     if "encoder" in name:
                #         yield name,p
                # else:
                #     continue
                yield name, p

    def named_parameters(self, memo=None, prefix='', recurse=True):
        return self.named_params(self, memo=memo, prefix=prefix)

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False, shared_params=[]):
        only_shared=False
        if len(shared_params)>0:
            only_shared=True
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                if src.is_sparse:
                    src=src.to_dense()
                name_t, param_t = tgt
                if only_shared:
                    if not name_t in shared_params:
                        continue
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                #print("updating: %s"%name_t)
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is None:
                    print(name_t)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        #print("name: %s"%name)
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
           # print(name)

            self.set_param(self, name, param.detach_().data)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaSequential(MetaModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    To make it easier to understand, here is a small example::
        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(MetaSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(MetaSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.has_bias = False
        self.in_features= ignore.in_features
        self.out_features=ignore.out_features
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        if ignore.bias is not None:
            self.has_bias = True
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.bias=None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        if self.has_bias:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MetaLSTMCell(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.LSTMCell(*args, **kwargs)


        self.register_buffer('weight_ih', to_var(ignore.weight_ih.data, requires_grad=True))
        self.register_buffer('weight_hh', to_var(ignore.weight_hh.data, requires_grad=True))
        self.register_buffer('bias_ih', to_var(ignore.bias_ih.data, requires_grad=True))
        self.register_buffer('bias_hh', to_var(ignore.bias_hh.data, requires_grad=True))

        self.input_size=ignore.input_size
        self.hidden_size=ignore.hidden_size

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')

        return _VF.lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )



    def named_leaves(self):
        return [('weight_ih', self.weight_ih), ('weight_hh', self.weight_hh),
                ('bias_ih', self.bias_ih), ('bias_hh', self.bias_hh)]

class MetaEmbeddingBase(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Embedding(*args, **kwargs)
        self.padding_idx=ignore.padding_idx
        self.max_norm=ignore.max_norm
        self.norm_type = ignore.norm_type
        self.scale_grad_by_freq=ignore.scale_grad_by_freq
        self.sparse=ignore.sparse
        self.num_embeddings=ignore.num_embeddings
        self.embedding_dim=ignore.embedding_dim
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

    def named_leaves(self):
        return [('weight', self.weight)]

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        return self.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def embedding(self, input, weight, padding_idx=None, max_norm=None, norm_type=2,
                  scale_grad_by_freq=False, sparse=False):

        input = input.contiguous()
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < weight.size(0), 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -weight.size(0), 'Padding_idx must be within num_embeddings'
                padding_idx = weight.size(0) + padding_idx
        elif padding_idx is None:
            padding_idx = -1
        if max_norm is not None:
            with torch.no_grad():
                torch.embedding_renorm_(weight, input, max_norm, norm_type)
        return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding
