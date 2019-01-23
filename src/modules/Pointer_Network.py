# encoding: utf-8
"""
@author: Meefly
@contact: admin@meijiex.vip
@ref: https://github.com/shiretzet/PointerNet

@version: 1.0
@file: Pointer_Network.py
@time: 2019年1月16日 14:39:14

The original algorithm is modified in the following two aspects:
1. The input don't have to be embedded
2. Only need 2 pointers not input.size(1)
"""
import torch
import torch.nn as nn
import logging
from torch.nn import Parameter
from allennlp.nn import util
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules import TimeDistributed
from overrides import overrides

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class PointerNetEncoder(nn.Module):
    """
    PointerNetEncoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate PointerNetEncoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNetEncoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        """
        PointerNetEncoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, batch_size):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """
        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = torch.softmax(att, dim=-1)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class PointerNetDecoder(nn.Module):
    """
    PointerNetDecoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate PointerNetDecoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PointerNetDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)

        self._ptr_layer_1 = TimeDistributed(torch.nn.Linear(embedding_dim * 2,
                                                            embedding_dim, bias=False))
        self._ptr_layer_2 = TimeDistributed(torch.nn.Linear(hidden_dim,
                                                            embedding_dim))
        self._ptr_layer_3 = TimeDistributed(torch.nn.Linear(embedding_dim, 1))

    def forward(self,
                embedded_inputs,    # input_sequence
                hidden,             # lstm hidden of input_sequence
                context,            # output of lstm
                decoder_input,
                passages_mask):
        """
        PointerNetDecoder - Forward-pass
        Return
        ------
        span_start_logits, span_end_logits
        """

        batch_size = embedded_inputs.size(0)
        passage_length = embedded_inputs.size(1)

        # (batch_size, passage_length)
        mask = self.mask.view(1, 1).repeat(batch_size, passage_length)
        # -inf
        self.att.init_inf(mask.size())

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # (, passage_length)
        logger.debug("embedded_inputs.size %s" % str(embedded_inputs.size()))
        logger.debug("start_hidden[0].size %s" % str(hidden[0].size()))
        logger.debug("start_hidden[1].size %s" % str(hidden[1].size()))
        F = torch.tanh(self._ptr_layer_1(torch.cat([embedded_inputs,
                                                    embedded_inputs.new_zeros(
                                                        embedded_inputs.size())], dim=-1)) +
                       self._ptr_layer_2(hidden[0].unsqueeze(1)).repeat(1, passage_length, 1))
        logger.debug("F.size %s" % str(F.size()))
        # linear with bias
        span_start_logits = self._ptr_layer_3(F).squeeze()
        logger.debug("span_start_logits.size %s" % str(span_start_logits.size()))
        span_start_probs = util.masked_softmax(span_start_logits, passages_mask)
        decoder_input = util.weighted_sum(embedded_inputs, span_start_probs)
        logger.debug("decoder_input.size %s" % str(decoder_input.size()))
        h_t, c_t, _ = step(decoder_input, hidden)
        hidden = (h_t, c_t)
        logger.debug("end_hidden[0].size %s" % str(hidden[0].size()))
        logger.debug("end_hidden[1].size %s" % str(hidden[1].size()))
        # end
        F = torch.tanh(self._ptr_layer_1(torch.cat([embedded_inputs,
                                                    embedded_inputs.new_zeros(
                                                        embedded_inputs.size())], dim=-1)) +
                       self._ptr_layer_2(hidden[0].unsqueeze(1)).repeat(1, passage_length, 1))
        span_end_logits = self._ptr_layer_3(F).squeeze()
        # span_end_probs = util.masked_softmax(span_end_logits, passages_mask)
        return span_start_logits, span_end_logits


@Seq2SeqEncoder.register("PointerNet")
class PointerNet(Seq2SeqEncoder):
    """
    Pointer-Net
    """

    def __init__(self, input_size: int,
                 hidden_dim: int,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional=False):
        """
        Initiate Pointer-Net

        :param int input_size: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """
        super(PointerNet, self).__init__()
        self.bidir = bidirectional
        self._encoder = PointerNetEncoder(input_size,
                                          hidden_dim,
                                          lstm_layers,
                                          dropout,
                                          bidirectional)
        self._decoder = PointerNetDecoder(input_size, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(input_size), requires_grad=False)
        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    @overrides
    def forward(self, embedded_inputs: torch.Tensor, passages_mask: torch.Tensor):
        """
        PointerNet - Forward-pass

        Parameters
        ----------
        embedded_inputs
            Shape(batch_size * num_passages, passage_length, embedding_size)
        Return
        ------
        span_start_logits, span_end_logits
        """
        batch_size = embedded_inputs.size(0)
        # uniform initialize
        fake_inputs = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # zeros initialize
        encoder_hidden0 = self._encoder.init_hidden(batch_size)
        # use lstm to encode
        encoder_outputs, encoder_hidden = self._encoder(embedded_inputs,
                                                        encoder_hidden0)
        # pdb.set_trace()
        if self.bidir:
            hidden0 = (torch.cat(tuple(encoder_hidden[0][-2:]), dim=-1),
                       torch.cat(tuple(encoder_hidden[1][-2:]), dim=-1))
        else:
            hidden0 = (encoder_hidden[0][-1],
                       encoder_hidden[1][-1])
        return self._decoder(embedded_inputs,
                             hidden0,            # hidden
                             encoder_outputs,    # context
                             fake_inputs,     # decoder_input
                             passages_mask)
