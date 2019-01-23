"""
This module implements the Pointer Network for selecting answer spans, as described in:
https://openreview.net/pdf?id=B1-q5Pqxl
begin time: 2019年1月23日 19:15:56
"""
import torch

class PointerNetLSTMCell(torch.nn.LSTMCell):
    """
    Implements the Pointer Network Cell
    """

    def __init__(self, input_emb_size,
                 hidden_size,
                 context_to_point):
        super(PointerNetLSTMCell, self).__init__(input_emb_size, hidden_size)
        self.context_to_point = context_to_point
        self.hidden_size = hidden_size
        self.device = context_to_point.device
        self.to(self.device)
        # layers
        self._fc = torch.nn.Linear(input_emb_size, hidden_size)
        self._fc_m = torch.nn.Linear(hidden_size, hidden_size)
        self._fc_att = torch.nn.Linear(hidden_size, 1)
        self._fc.to(self.device)
        self._fc_m.to(self.device)
        self._fc_att.to(self.device)
        self.fc_context = self._fc(context_to_point)

    def forward(self, inputs, state):
        if state is None:
            state = torch.zeros((inputs.size(0), self.hidden_size)).to(self.device)
            state = (state, state)
        (c_prev, m_prev) = state
        U = torch.tanh(self.fc_context +
                       self._fc_m(m_prev).unsqueeze(1))
        # Shape: (batch_size, passage_length, 1)
        logits = self._fc_att(U)
        # Shape: (batch_size, passage_length, 1)
        scores = torch.softmax(logits, dim=1)
        # Shape: (batch_size, input_emb_size)
        attended_context = torch.sum(self.context_to_point * scores, dim=1)

        lstm_out, lstm_state = super(PointerNetLSTMCell, self).forward(attended_context, state)
        return logits.squeeze(-1), lstm_state


class PointerNetDecoder(torch.nn.Module):
    """
    Implements the Pointer Network
    """

    def __init__(self, input_emb_size: int,
                 hidden_size: int,
                 init_with_question: bool=False):
        '''
        init_with_question: if set to be true,
                         we will use the question_vectors to init the state of Pointer Network
        '''
        super(PointerNetDecoder, self).__init__()
        # parameter
        self.hidden_size = hidden_size
        self.init_with_question = init_with_question
        self.input_emb_size = input_emb_size
        # attend_pooling
        self._fc_att_data = torch.nn.Linear(input_emb_size, hidden_size)
        self._fc_att_hidden = torch.nn.Linear(hidden_size, hidden_size)
        self._fc_att = torch.nn.Linear(hidden_size, 1)
        self._fc_att_rep = torch.nn.Linear(input_emb_size, hidden_size)

    def forward(self, passage_vectors, question_vectors):
        """
        Use Pointer Network to compute the probabilities of each position
        to be start and end of the answer
        Args:
            passage_vectors: the encoded passage vectors
            question_vectors: the encoded question vectors
        Returns:
            the probs of evary position to be start and end of the answer
        """
        batch_size = passage_vectors.size(0)
        device = passage_vectors.device
        # Shape (batch_size, 2, 1)
        fake_inputs = torch.zeros((batch_size, 2, 1)).to(device)
        if self.init_with_question:
            random_attn_vector = torch.autograd.Variable(torch.rand((1, self.hidden_size)),
                                                         requires_grad=True)
            # Shape: (batch_size, input_emb_size)
            pooled_question_vectors = self.attend_pooling(question_vectors, random_attn_vector)
            # Shape: (batch, hidden_size)
            pooled_question_rep = self._fc_att_rep(pooled_question_vectors)
            # LSTM state (c, h)
            init_state = (pooled_question_rep, pooled_question_rep)
        else:
            init_state = None
        # lstm forward
        fw_outputs = []
        self.fw = PointerNetLSTMCell(self.input_emb_size, self.hidden_size, passage_vectors)
        for i in range(2):
            hx, cx = self.fw(fake_inputs[:, i, :], init_state)
            fw_outputs.append(hx)
        # lstm backward
        bw_outputs = []
        self.bw = PointerNetLSTMCell(self.input_emb_size, self.hidden_size, passage_vectors)
        for i in range(2):
            hx, cx = self.bw(fake_inputs[:, i, :], init_state)
            bw_outputs.append(hx)
        # results
        start_logits = (fw_outputs[0] + bw_outputs[1]) / 2
        end_logits = (fw_outputs[1] + bw_outputs[0]) / 2
        return start_logits, end_logits

    def attend_pooling(self, pooling_vectors, ref_vector):
        """
        Applies attend pooling to a set of vectors according to a reference vector.
        Args:
            pooling_vectors: the vectors to pool
            ref_vector: the reference vector
            hidden_size: the hidden size for attention function
            scope: score name
        Returns:
            the pooled vector
        """
        U = torch.tanh(self._fc_att_data(pooling_vectors) +
                       self._fc_att_hidden(ref_vector.view(1, 1, self.hidden_size)))
        # Shape: (batch_size, passage_length, 1)
        logits = self._fc_att(U)
        # Shape: (batch_size, passage_length, 1)
        scores = torch.softmax(logits, dim=1)
        # Shape: (batch_size, input_emb_size)
        pooled_vector = torch.sum(pooling_vectors * scores, dim=1)
        return pooled_vector
