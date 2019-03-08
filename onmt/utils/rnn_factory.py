"""
 RNN tools
"""
import torch.nn as nn
import onmt.models

def  rnn_factory(rnn_type, main_rnn=None, mtl_opt=None, module_type=None, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.models.sru.SRU(**kwargs)
    elif rnn_type == "sharedLSTM":
        # Shared_layer_GRU doesn't support PackedSequence
        no_pack_padded_seq = True
        rnn = onmt.models.mtl_rnn.SharedLayerLSTM(main_rnn=main_rnn, mtl_opt=mtl_opt, module_type=module_type, **kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq
