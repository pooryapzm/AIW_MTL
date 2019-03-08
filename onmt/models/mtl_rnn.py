import torch
import torch.nn as nn
import numbers
import warnings
from torch.nn.utils.rnn import PackedSequence
from onmt.meta_modules import MetaLSTMCell, MetaModule

# TODO: bidirectional LSTM is not working, check it!

#class SharedLayerLSTM(nn.Module):
class SharedLayerLSTM(MetaModule):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=False, batch_first=False,
                 dropout=0, bidirectional=False, mtl_opt=None, main_rnn=None, module_type="encoder"):
        super(SharedLayerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.module_type = module_type

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers_fwd = nn.ModuleList()
        if bidirectional:
            self.layers_bwd = nn.ModuleList()
        else:
            self.layers_bwd = None

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
        if main_rnn is None:
            main_layers_fwd = None
            main_layers_bwd = None
            n_shared_layers = 0
        else:
            main_layers_fwd = main_rnn.rnn.layers_fwd
            main_layers_bwd = main_rnn.rnn.layers_bwd

        is_share_top = True
        #if mtl_opt is not None and mtl_opt.mtl:
        if module_type == "encoder":
            n_shared_layers = int(mtl_opt.shared_enc_layers)
            if mtl_opt.share_encoder_bottom:
                is_share_top = False
        elif module_type == "decoder":
            n_shared_layers = int(mtl_opt.shared_dec_layers)
            if mtl_opt.share_decoder_bottom:
                is_share_top = False
        else:
            raise ValueError("Module_type (%s) is not recognized" % module_type)

        for i in range(num_layers):
            is_layer_shared = False
            if main_layers_fwd is not None:
                if is_share_top:
                    if i >= num_layers - n_shared_layers:
                        is_layer_shared = True
                else:
                    if i < n_shared_layers:
                        is_layer_shared = True

            if not is_layer_shared:
                self.layers_fwd.append(MetaLSTMCell(input_size, hidden_size))
                #self.layers_fwd.append(nn.LSTMCell(input_size, hidden_size))
                if self.bidirectional:
                    self.layers_bwd.append(MetaLSTMCell(input_size, hidden_size))
                    #self.layers_bwd.append(nn.LSTMCell(input_size, hidden_size))
                input_size = hidden_size
            else:
                self.layers_fwd.append(main_layers_fwd[i])
                if self.bidirectional:
                    self.layers_bwd.append(main_layers_bwd[i])
                # the following line should be here, otherwise if we share the first layer,
                #  then the input-size of the unit remains the real input size
                input_size = hidden_size

    def get_layers_cells(self):
        return self.layers_fwd, self.layers_bwd

    def forward(self, input, hidden=None):
        if self.module_type.startswith("decoder"):
            hidden = hidden[0]
            if len(input.size()) == 2:
                input = input.unsqueeze(0)
        is_packed = isinstance(input, PackedSequence)
        # info: at the moment it doesn't support PackedSequence
        input_lengths = None
        if is_packed:
            input, input_lengths = nn.utils.rnn.pad_packed_sequence(input, self.batch_first)

        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hidden is None:
            hidden = input.new_zeros(self.num_layers * self.num_directions,
                                     max_batch_size, self.hidden_size,
                                     requires_grad=False)

        if self.batch_first:
            # do the following transformation.(#batch, seq, hidden_size) -> (seq, #batch, hidden_size)
            input = input.transpose(0, 1)
        steps = input.size(0)
        output = []
        if self.bidirectional:
            hidden_fwd = hidden.view(self.num_layers, 2, max_batch_size, self.hidden_size)[:, 0, :, :]
            cell_fwd = hidden.view(self.num_layers, 2, max_batch_size, self.hidden_size)[:, 0, :, :]
            hidden_bwd = hidden.view(self.num_layers, 2, max_batch_size, self.hidden_size)[:, 1, :, :]
            cell_bwd = hidden.view(self.num_layers, 2, max_batch_size, self.hidden_size)[:, 1, :, :]
        else:
            hidden_fwd = hidden
            cell_fwd = hidden

        for step in range(steps):  # each step process 1 word for all of the sequences in the batch
            current_input_fwd = input[step][:]
            current_input_bwd = input[-step - 1][:]

            h_1_fwd = []
            h_1_bwd = []
            c_1_fwd = []
            c_1_bwd = []
            for i, layer_fwd in enumerate(self.layers_fwd):

                h_1_i_fwd, c_1_i_fwd = layer_fwd(current_input_fwd, (hidden_fwd[i][:], cell_fwd[i][:]))
                current_input_fwd = h_1_i_fwd
                if i + 1 != self.num_layers:
                    current_input_fwd = self.dropout(current_input_fwd)
                if self.bidirectional:
                    layer_bwd = self.layers_bwd[i]
                    h_1_i_bwd, c_1_i_bwd = layer_bwd(current_input_bwd, (hidden_bwd[i][:], cell_bwd[i][:]))
                    current_input_bwd = h_1_i_bwd
                    if i + 1 != self.num_layers:
                        current_input_bwd = self.dropout(current_input_bwd)
                h_1_fwd += [h_1_i_fwd.unsqueeze(0)]
                c_1_fwd += [c_1_i_fwd.unsqueeze(0)]
                if self.bidirectional:
                    h_1_bwd += [h_1_i_bwd.unsqueeze(0)]
                    c_1_bwd += [c_1_i_bwd.unsqueeze(0)]
            hidden_fwd = torch.cat(h_1_fwd)
            cell_fwd = torch.cat(c_1_fwd)
            if self.bidirectional:
                hidden_bwd = torch.cat(h_1_bwd)
                cell_bwd = torch.cat(c_1_bwd)
                output += [torch.cat([current_input_fwd, current_input_bwd], dim=1).unsqueeze(1)]
            else:
                output += [current_input_fwd.unsqueeze(1)]
        if self.bidirectional:
            # Twist the hidden states of forward and backward RNNs
            h = []
            for i in range(self.num_layers):
                h += [h_1_fwd[i]]
                h += [h_1_bwd[i]]
            hidden = torch.cat(h)
        else:
            hidden = hidden_fwd
        output = torch.cat(output, dim=1)

        output = output.transpose(0, 1)

        if is_packed:
            output = nn.utils.rnn.pack_padded_sequence(output, input_lengths, self.batch_first)
        if self.module_type == "encoder":
            return output, hidden
        else:
            return output[0], (hidden,)
