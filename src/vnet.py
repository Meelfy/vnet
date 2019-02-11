# encoding: utf-8
"""
@author: Meefly
@contact: admin@meijiex.vip

@version: 1.0
@file: vnet.py
@time: 2018年12月30日 21:34:21

这一行开始写关于本文件的说明与解释
"""
import logging
from typing import Any, Dict, List, Optional
import random

import torch
from torch.nn.functional import nll_loss
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from .MsmarcoRouge import MsmarcoRouge
from .DureaderBleu import DureaderBleu
from .modules.Pointer_Network import PointerNet
# from .modules.pointerNetwork import PointerNetDecoder
from .modules.ElasticHighway import ElasticHighway
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


@Model.register('vnet')
class VNet(Model):
    """
    This class implements Yizhong Wang's Multi-Passage Machine Reading Comprehension with Cross-Passage
    Answer Verification (https://arxiv.org/abs/1805.02220)
    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 highway_embedding_size: int,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 # match_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 pointer_net: PointerNet,
                 span_end_lstm: Seq2SeqEncoder,
                 language: str = 'en',
                 ptr_dim: int = 200,
                 dropout: float = 0.2,
                 loss_ratio: float = 0.3,
                 max_num_passages: int = 5,
                 max_num_character: int = 4,
                 max_passage_len: int = 4,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        # self._span_end_encoder = span_end_lstm
        self.language = language
        self.loss_ratio = loss_ratio
        self.max_num_character = max_num_character
        self.relu = torch.nn.ReLU()
        self.max_num_passages = max_num_passages
        self.max_passage_len = max_passage_len
        self.ptr_dim = ptr_dim
        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(ElasticHighway(text_field_embedder.get_output_dim(),
                                                             highway_embedding_size,
                                                             num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = DotProductMatrixAttention()
        self._modeling_layer = modeling_layer
        modeling_dim = modeling_layer.get_output_dim()
        encoding_dim = phrase_layer.get_output_dim()

        # self._match_layer = match_layer
        self._ptr_layer_1 = TimeDistributed(torch.nn.Linear(encoding_dim * 4 +
                                                            modeling_dim, 1))
        self._ptr_layer_2 = TimeDistributed(torch.nn.Linear(encoding_dim * 4 +
                                                            modeling_dim, 1))
        # self._naive_layer_1 = TimeDistributed(torch.nn.Linear(highway_embedding_size, 1))
        # self._naive_layer_2 = TimeDistributed(torch.nn.Linear(highway_embedding_size, 1))

        self._content_layer_1 = TimeDistributed(torch.nn.Linear(encoding_dim * 4 +
                                                                modeling_dim, ptr_dim))
        self._content_layer_2 = TimeDistributed(torch.nn.Linear(ptr_dim, 1))

        self._passages_matrix_attention = matrix_attention_layer

        self._pointer_net = pointer_net
        # self._pointer_net_decoder = PointerNetDecoder(encoding_dim * 4 +
        #                                               modeling_dim,
        #                                               ptr_dim)

        self._passage_predictor = TimeDistributed(torch.nn.Linear(self.max_num_passages, 1))

        self._start_h_embedding = torch.nn.Parameter(data=torch.zeros(1, 1, 1).float(),
                                                     requires_grad=True)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._rouge_metrics = MsmarcoRouge()
        self._bleu_metrics = DureaderBleu()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passages: List[Dict[str, torch.LongTensor]],
                spans_start: List[List[torch.IntTensor]] = None,
                spans_end: List[List[torch.IntTensor]] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passages : List[Dict[str, torch.LongTensor]]
            From a ``ListField[TextField]``.  The model assumes that one question corresponds to
            more than one article, and each article can correspond to more than one answer. And it
            will predict the beginning and ending positions of the answer within the passage.
        spans_start : ``List[List[torch.IntTensor]]``, optional
            From an ``ListField[ListField[IndexField]]``.  This is one of the things we are trying
            to predict - the beginning position of the answer with the passage.  This is an
            `inclusive` token index. If this is given, we will compute a loss that gets included in
            the output dictionary.
        spans_end : ``List[List[torch.IntTensor]]``, optional
            From an ``ListField[ListField[IndexField]]``.  This is one of the things we are trying
            to predict - the ending position of the answer with the passage.  This is an `inclusive`
            token index. If this is given, we will compute a loss that gets included in the output
            dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official MSMARCO bleu-1 and rouge-L evaluation script.
            The length of this list should be the batch size, and each dictionary should have the
            keys ``qid``, ``original_passages``, ``question_tokens`` , ``passage_tokens``, and
            ``passages_offsets``.
        Returns
        -------
        An output dictionary consisting of:
        spans_start_logits : List[torch.FloatTensor]
            A tensor of shape ``(batch_size, num_passages, passage_length)`` representing unnormalized
            log probabilities of the span start position.
        spans_start_probs : List[torch.FloatTensor]
            The result of ``softmax(spans_start_logits)``.
        spans_end_logits : List[torch.FloatTensor]
            A tensor of shape ``(batch_size, num_passages, passage_length)`` representing unnormalized
            log probabilities of the span end position (inclusive).
        spans_end_probs : List[torch.FloatTensor]
            The result of ``softmax(span_end_logits)``.
        best_passage_id: torch.IntTensor
            The idx of the best answer source article.        in range(0,num_passages)
        best_span : List[torch.IntTensor]
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, num_passages, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        # ---------------------------------------
        # Part One: Question and Passage Modeling
        # ---------------------------------------
        device = passages['tokens'].device
        # passages['token_characters']
        #   torch.Size([batch_size, num_passages, passage_length, num_characters])
        # passages['tokens']
        #   torch.Size([batch_size, num_passages, passage_length])
        # shape(passages_batch_size=num_passages*batch_size, question_length, question_embedding_size )
        batch_size, num_passages, passage_length = passages['tokens'].size()
        pad_size_p_length = self.max_passage_len - passage_length
        pad_size_p_length = max(pad_size_p_length, 0)
        # shape(batch_size*num_passages, passage_length, num_characters)
        batch_passages = {}
        if 'token_characters' in passages:
            num_characters = passages['token_characters'].size(-1)
            batch_passages['token_characters'] = passages['token_characters'].view(batch_size * num_passages,
                                                                                   passage_length,
                                                                                   num_characters)
            batch_passages['token_characters'] = batch_passages['token_characters'][:, :,
                                                                                    :self.max_num_character]
            # pad for glyph
            pad_size_char = self.max_num_character - batch_passages['token_characters'].size(-1)
            pad_size_char = max(pad_size_char, 0)
            # pad for ptr-net
            batch_passages['token_characters'] = F.pad(batch_passages['token_characters'],
                                                       (0, pad_size_char, 0, pad_size_p_length),
                                                       'constant',
                                                       0.0)
        # shape(batch_size*num_passages, passage_length)
        passage_length = self.max_passage_len
        batch_passages['tokens'] = F.pad(passages['tokens'],
                                         (0, pad_size_p_length),
                                         'constant',
                                         0.0)
        batch_passages['tokens'] = batch_passages['tokens'].view(-1, passage_length)
        # shape(batch_size*num_passages, passage_length, embedding_dim)
        if "_token_embedders" in dir(self._text_field_embedder) \
                and 'token_characters' in self._text_field_embedder._token_embedders.keys()\
                and 'using_glyph' in dir(self._text_field_embedder._token_embedders['token_characters']):
            embedded_passages, glyph_loss_p = self._text_field_embedder(batch_passages)
            embedded_passages = self._highway_layer(embedded_passages)
        else:
            embedded_passages = self._highway_layer(self._text_field_embedder(batch_passages))
        embedding_dim = embedded_passages.size(-1)

        # shape(batch_size, question_length, num_characters)
        questions = {}
        batch_size, question_length = question['tokens'].size()
        if 'token_characters' in question:
            num_characters = question['token_characters'].size(-1)
            questions['token_characters'] = question['token_characters'].repeat(1, num_passages, 1)\
                                                                        .view(batch_size * num_passages,
                                                                              question_length,
                                                                              num_characters)
            questions['token_characters'] = questions['token_characters'][:, :, :self.max_num_character]
            pad_size_char = self.max_num_character - questions['token_characters'].size(-1)
            questions['token_characters'] = F.pad(questions['token_characters'],
                                                  (0, pad_size_char),
                                                  'constant',
                                                  0.0)
        questions['tokens'] = question['tokens'].repeat(1, num_passages).view(-1, question_length)

        if "_token_embedders" in dir(self._text_field_embedder) \
                and 'token_characters' in self._text_field_embedder._token_embedders.keys()\
                and 'using_glyph' in dir(self._text_field_embedder._token_embedders['token_characters']):
            embedded_question, glyph_loss_q = self._text_field_embedder(question)
            embedded_question = self._highway_layer(embedded_question)
        else:
            embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedding_size = embedded_question.size(-1)
        # shape(num_passages*batch_size, question_length, embedding_size)
        embedded_questions = embedded_question.repeat(1, num_passages, 1)\
                                              .view(-1, question_length, embedding_size)
        assert embedded_questions.size(0) == embedded_passages.size(0)
        assert torch.eq(embedded_questions[0, :, :], embedded_questions[num_passages - 1, :, :]).all()
        # shape(num_passages*batch_size, question_length)
        questions_mask = util.get_text_field_mask(questions).float()
        # shape(num_passages*batch_size, passage_length)
        passages_mask = util.get_text_field_mask(batch_passages).float()

        # shape(num_passages*batch_size, question_length)
        questions_lstm_mask = questions_mask if self._mask_lstms else None
        # shape(num_passages*batch_size, passage_length)
        passages_lstm_mask = passages_mask if self._mask_lstms else None

        # encoded_question
        #     torch.Size([num_passages*batch_size, question_length, phrase_layer_encoding_dim])
        encoded_questions = self._dropout(self._phrase_layer(embedded_questions, questions_lstm_mask))
        phrase_layer_encoding_dim = encoded_questions.size(-1)
        # encoded_passages
        #     torch.Size([num_passages*batch_size, passage_length, phrase_layer_encoding_dim])
        encoded_passages = self._dropout(self._phrase_layer(embedded_passages, passages_lstm_mask))
        # Shape: (num_passages*batch_size, passage_length, question_length)
        passages_questions_similarity = self._matrix_attention(encoded_passages, encoded_questions)
        # Shape: (num_passages*batch_size, passage_length, question_length)
        passages_questions_attention = util.masked_softmax(passages_questions_similarity, questions_mask)

        # Shape: (num_passages*batch_size, passage_length, phrase_layer_encoding_dim)
        passages_questions_vectors = util.weighted_sum(encoded_questions, passages_questions_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passages_questions_similarity,
                                                       questions_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size * num_passages, passage_length)
        questions_passages_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size * num_passages, passage_length)
        questions_passages_attention = util.masked_softmax(questions_passages_similarity, passages_mask)
        # Shape: (batch_size * num_passages, phrase_layer_encoding_dim)
        questions_passages_vector = util.weighted_sum(encoded_passages, questions_passages_attention)
        # Shape: (batch_size * num_passages, passage_length, phrase_layer_encoding_dim)
        tiled_questions_passages_vector = questions_passages_vector.unsqueeze(1)\
                                                                   .expand(batch_size * num_passages,
                                                                           passage_length,
                                                                           phrase_layer_encoding_dim)

        # Shape: (batch_size, passage_length, phrase_layer_encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passages,
                                          passages_questions_vectors,
                                          encoded_passages * passages_questions_vectors,
                                          encoded_passages * tiled_questions_passages_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passages_lstm_mask))

        # ------------------------------------
        # Part Two: Answer Boundary Prediction
        # ------------------------------------
        # # Shape: (num_passages*batch_size, passage_length, phrase_layer_encoding_dim)
        # match_passages_vector = self._dropout(self._match_layer(passages_questions_vectors,
        #                                                         passages_lstm_mask))
        # Shape: (batch_size * num_passages, passage_length, encoding_dim * 4 + modeling_dim))
        match_passages_vector = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        match_size = match_passages_vector.size(-1)
        # LSTM
        # match_passages_vector = self._dropout(self._match_layer(
        #     match_passages_vector, torch.ones(match_passages_vector.size()[:2])
        #     .to(match_passages_vector.device)))

        # PointerNet
        # match_passages_vector = match_passages_vector.view(batch_size, -1, match_size)
        span_start_logits, span_end_logits = self._pointer_net(match_passages_vector, passages_mask)
        # span_start_logits = span_start_logits.view(batch_size * num_passages, passage_length)
        # span_end_logits = span_end_logits.view(batch_size * num_passages, passage_length)
        # match_passages_vector = match_passages_vector.view(batch_size * num_passages,
        #                                                    passage_length,
        #                                                    match_size)
        # span_start_logits, span_end_logits = self._pointer_net_decoder(match_passages_vector,
        #                                                                encoded_questions)
        # span_start_logits = self._ptr_layer_1(match_passages_vector).squeeze()
        # span_end_logits = self._ptr_layer_2(match_passages_vector).squeeze()

        # span_start_logits = self._naive_layer_1(embedded_passages).squeeze()
        # span_end_logits = self._naive_layer_2(embedded_passages).squeeze()
        # Shape: (num_passages*batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passages_mask)
        span_end_probs = util.masked_softmax(span_end_logits, passages_mask)

        # -----------------------------------
        # Part Three: Answer Content Modeling
        # -----------------------------------
        # shape(num_passages*batch_size, passage_length)
        p = torch.sigmoid(self._content_layer_2(self.relu(self._content_layer_1(
            match_passages_vector)))).squeeze(-1)

        # embedded_passages shape(batch_size*num_passages, passage_length, embedding_dim)
        # get answers candidates
        # shape(num_passages*batch_size, passage_length)
        # prob_p = self.get_prob_map(span_start_probs, span_end_probs)
        # # prob_p = prob_p.clamp(0, 1)
        # embedded_answers_candidates = embedded_passages *\
        #     prob_p.unsqueeze(-1).repeat(1, 1, embedding_dim)
        # embedded_answers_candidates = embedded_passages
        # shape(num_passages*batch_size, embedding_dim)
        r = util.weighted_sum(embedded_passages, p)

        # ---------------------------------------------
        # Part Four:  Cross-Passage Answer Verification
        # ---------------------------------------------
        # shape(batch_size, num_passages, embedding_dim)
        batch_r = r.view(batch_size, num_passages, embedding_dim)
        mask_diagonal_zero = (1 - torch.eye(num_passages, device=device))
        # shape(batch_size, num_passages, num_passages)
        passages_self_similarity = self._matrix_attention(batch_r, batch_r) * mask_diagonal_zero
        # shape(batch_size, num_passages, num_passages)
        passages_self_attention = torch.softmax(passages_self_similarity, dim=-1)
        # shape(batch_size, num_passages, embedding_dim)
        attention_batch_r = util.weighted_sum(batch_r, passages_self_attention)

        # shape(batch_size, num_passages, num_passages)
        g = self._passages_matrix_attention(batch_r, attention_batch_r)
        pad_size = self.max_num_passages - g.size(-1)
        g = F.pad(g, (0, pad_size, 0, pad_size), 'constant', 0.0)
        # shape(batch_size, num_passages)
        passages_verify = self._passage_predictor(g).squeeze(-1)
        passages_verify = passages_verify[:, :num_passages]
        # if self.training and random.randint(1, 10) != 1.1:
        #     best_span = None
        # else:
        best_span = self.get_best_span(span_start_probs.view(batch_size, num_passages, -1),
                                       span_end_probs.view(batch_size, num_passages, -1),
                                       util.masked_softmax(p, passages_mask).view(batch_size,
                                                                                  num_passages,
                                                                                  -1),
                                       torch.softmax(passages_verify, dim=-1))
        output_dict = {'best_span': best_span,
                       'span_start_logits': span_start_logits.view(batch_size, num_passages, -1),
                       'span_start_probs': span_start_probs.view(batch_size, num_passages, -1),
                       'span_end_logits': span_end_logits.view(batch_size, num_passages, -1),
                       'span_end_probs': span_end_probs.view(batch_size, num_passages, -1)}

        if spans_start is not None:
            # span_start_probs shape(num_passages*batch_size, passage_length)
            # spans_start shape(batch_size, num_passages, 1)
            # then shape(batch_size*num_passages, 1)
            spans_start = spans_start.view(batch_size * num_passages, 1)
            spans_end = spans_end.view(batch_size * num_passages, 1)

            spans_start.clamp_(-1, passage_length - 1)
            spans_end.clamp_(-1, passage_length - 1)
            # loss_Boundary = nll_loss(torch.log_softmax(span_start_logits, dim=-1),
            #                          spans_start.squeeze(-1), ignore_index=-1)
            # loss_Boundary += nll_loss(torch.log_softmax(span_end_logits, dim=-1),
            #                           spans_end.squeeze(-1), ignore_index=-1)
            loss_Boundary = nll_loss(util.masked_log_softmax(span_start_logits, passages_mask),
                                     spans_start.squeeze(-1), ignore_index=-1)
            loss_Boundary += nll_loss(util.masked_log_softmax(span_end_logits, passages_mask),
                                      spans_end.squeeze(-1), ignore_index=-1)
            loss_Boundary = loss_Boundary / 2
            eps = 1e-7
            p = p.clamp(eps, 1. - eps)
            ground_truth_p = self.span_idx_to_01_mask(spans_start, spans_end, p.size(-1))
            loss_Content = torch.sum(torch.log(p) * ground_truth_p)
            loss_Content += torch.sum(torch.log(1 - p) * (1 - ground_truth_p))
            loss_Content = -loss_Content / max(passage_length * batch_size, 1)

            # shape(batch_size, num_passages)
            ground_truth_passages_verify = (spans_end != -1).float().to(device).view(batch_size,
                                                                                     num_passages)
            # sigmoid passage loss
            passages_verify = torch.softmax(passages_verify, dim=-1)
            passages_verify = passages_verify.clamp(eps, 1. - eps)
            loss_Verification = torch.log(passages_verify) * ground_truth_passages_verify
            # softmax passage loss
            # loss_Verification = torch.log_softmax(passages_verify, dim=-1) * ground_truth_passages_verify
            loss_Verification = -torch.sum(loss_Verification) /\
                max(torch.sum(ground_truth_passages_verify), 1)
            loss = loss_Boundary + 0.5 * loss_Content + 0.5 * loss_Verification
            if 'glyph_loss_q' in locals():
                logger.debug('glyph_loss_q: %.5f' % glyph_loss_q)
                loss += glyph_loss_q * self.loss_ratio
            if 'glyph_loss_p' in locals():
                logger.debug('glyph_loss_p: %.5f' % glyph_loss_p)
                loss += glyph_loss_p * self.loss_ratio
            # loss = loss_Boundary + 0.5 * loss_Verification
            # loss = loss_Boundary
            logger.debug('loss_Boundary: %.5f' % loss_Boundary)
            logger.debug('loss_Content: %.5f' % loss_Content)
            logger.debug('loss_Verification: %.5f' % loss_Verification)
            output_dict['loss'] = loss
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                if best_span is None:
                    continue
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passages']
                offsets = metadata[i]['passages_offsets']
                passage_id, start_idx, end_idx = tuple(best_span[i, :])
                # passage_id = max(0, min(passage_id, len(offsets) - 1))
                # clamp start_idx and end_idx to range(0, passage_length - 1)
                start_idx = max(0, min(start_idx, len(offsets[passage_id]) - 1))
                end_idx = max(0, min(end_idx, len(offsets[passage_id]) - 1))

                start_offset = offsets[passage_id][start_idx][0]
                end_offset = offsets[passage_id][end_idx][1]
                best_span_string = passage_str[passage_id][start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if self.language == 'zh':
                    answer_texts = list(set([' '.join(item)
                                             for sublist in answer_texts for item in sublist]))
                elif self.language == 'en':
                    answer_texts = list(set(answer_texts))
                if answer_texts:
                    if self.language == 'zh':
                        self._rouge_metrics(' '.join(best_span_string), answer_texts)
                        self._bleu_metrics(' '.join(best_span_string), answer_texts)
                    elif self.language == 'en':
                        self._rouge_metrics(best_span_string, answer_texts)
                        self._bleu_metrics(best_span_string, answer_texts)
                if spans_start is not None:
                    logger.debug('passage_id:%d, start_idx:%d, end_idx:%d' %
                                 (passage_id, start_idx, end_idx))
                    logger.debug("spans_start: {}".format(
                        ' '.join(map(str, spans_start.view(batch_size, num_passages)[i]
                                     .cpu().numpy()))))
                    logger.debug("spans_end: {}".format(
                        ' '.join(map(str, spans_end.view(batch_size, num_passages)[i]
                                     .cpu().numpy()))))
                logger.debug('Predict: %s' % output_dict['best_span_str'][-1])
                for ans in answer_texts:
                    if self.language == 'zh':
                        logger.debug('Truth: %s' % ans.replace(' ', ''))
                    elif self.language == 'en':
                        logger.debug('Truth: %s' % ans)
            if spans_start is not None:
                self._span_start_accuracy(span_start_probs.view(batch_size, num_passages, -1),
                                          spans_start.view(batch_size, num_passages),
                                          (spans_start.view(batch_size, num_passages) != -1))
                self._span_end_accuracy(span_end_probs.view(batch_size, num_passages, -1),
                                        spans_end.view(batch_size, num_passages),
                                        (spans_end.view(batch_size, num_passages) != -1))
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        output_dict['qids'] = [data['qid'] for data in metadata]
        return output_dict

    @staticmethod
    def get_best_span(span_start_probs: torch.Tensor,
                      span_end_probs: torch.Tensor,
                      content: torch.Tensor,
                      passages_verify: torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        span_start_probs: shape(batch_size, num_passages, passage_length)
        span_end_probs: shape(batch_size, num_passages, passage_length)
        content: shape(batch_size, num_passages, passage_length)
        passages_verify: shape(batch_size, num_passages)
        Return
        ------
        best_word_span: shape(batch_size, 3)
            3 for [best_passage_id, start, end]
        speedup
        -------
        old version
            CPU 4.49 s ± 58.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            GPU 4.47 s ± 63.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        new version
            CPU 5.75 s ± 14.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            GPU 61.9 ms ± 37.3 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        4.47 / 61.9 * 1000 = 72.2
        new version is a little hard to understand
        '''
        if span_start_probs.dim() != 3 or span_end_probs.dim() != 3:
            raise ValueError("Input shapes must be (batch_size, num_passages, passage_length)")
        batch_size, num_passages, passage_length = span_start_probs.size()
        device = span_start_probs.device
        content = content.view(batch_size * num_passages, passage_length)

        span_start_probs = span_start_probs.view(batch_size * num_passages, passage_length)
        span_end_probs = span_end_probs.view(batch_size * num_passages, passage_length)
        span_probs = torch.bmm(span_start_probs.unsqueeze(2), span_end_probs.unsqueeze(1))
        span_probs_mask = torch.triu(torch.ones((passage_length, passage_length),
                                                device=device))
        valid_span_probs = span_probs * span_probs_mask

        cumsum_content = content.unsqueeze(1).repeat(1, passage_length, 1) *\
            torch.triu(torch.ones((passage_length, passage_length), device=device))
        cumsum_content = torch.cumsum(cumsum_content, dim=-1) /\
            torch.cumsum(torch.triu(torch.ones((passage_length, passage_length), device=device)), dim=-1)
        cumsum_content[cumsum_content != cumsum_content] = 0.0

        best_spans = valid_span_probs * cumsum_content
        best_spans = best_spans.view(batch_size, num_passages, passage_length, passage_length) *\
            passages_verify.view((batch_size, num_passages, 1, 1))
        best_spans = best_spans.view(batch_size, num_passages * (passage_length ** 2)).argmax(-1)
        passage_idx = best_spans // (passage_length ** 2)
        span_start_idx = best_spans % (passage_length ** 2) // passage_length
        span_end_idx = best_spans % (passage_length ** 2) % passage_length
        span_start_probs = span_start_probs.view(batch_size, num_passages, passage_length)
        span_end_probs = span_end_probs.view(batch_size, num_passages, passage_length)
        content = content.view(batch_size, num_passages, passage_length)
        return torch.stack([passage_idx, span_start_idx, span_end_idx], dim=-1)

    @staticmethod
    def map_span_to_01(span_idx: torch.Tensor, shape: int) -> torch.Tensor:
        '''
        This func is map span_idx=[[0], [2], [1]], shape = 3 to
        [[1, 0, 0],
         [1, 1, 1],
         [1, 1, 0]]
        line i has (span_idx[i]+1)  zeors
        Parameters
        ----------
        span_idx shape(batch_size * num_passages, 1)
        '''
        device = span_idx.device
        if len(span_idx.size()) == 1:
            span_idx = span_idx.unsqueeze(1)
        return 1 - torch.cumsum(torch.eye(shape + 1, device=device)[span_idx][:, :shape, :shape],
                                dim=-1).squeeze(1)

    def span_idx_to_01_mask(self, span_start_idx, span_end_idx, shape):
        '''
        Example
        -------
        span_start_idx = tensor([1, -1, 3, 0, 2, 2])
        span_end_idx   = tensor([2, -1, 3, 0, 2, 2])
        shape = 4

        Return
        ------
        return tensor([[0., 1., 1., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 1.],
                       [1., 0., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 1., 0.]])
        '''
        res = self.map_span_to_01(span_end_idx + 1, shape) -\
            self.map_span_to_01(span_start_idx, shape)
        return res.clamp(0, 1)

    def get_prob_map(self, span_start_probs: torch.Tensor,
                     span_end_probs: torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        span_start_logits: shape(batch_size * num_passages, passage_length)
        span_end_logits: shape(batch_size * num_passages, passage_length)

        Return
        ------
        prob_p_mask: shape(batch_size * num_passages, passage_length)
        for one passage the return like [0, 0, 0, 1, 1, 1, 0, 0]
        [span_start: span_end] is 1 and other is 0
        '''
        if span_start_probs.dim() != 2 or span_end_probs.dim() != 2:
            raise ValueError("Input shapes must be (batch_size * num_passages, passage_length)")
        batch_size, passage_length = span_start_probs.size()
        device = span_start_probs.device
        probs_product = torch.bmm(span_start_probs.unsqueeze(2), span_end_probs.unsqueeze(1))
        probs_product_triu = probs_product * torch.triu(torch.ones((passage_length, passage_length),
                                                                   device=device)).unsqueeze(0)
        res = probs_product_triu.view(batch_size, -1).argmax(-1)
        span_start_idx = res // passage_length
        span_end_idx = res % passage_length
        prob_p = self.span_idx_to_01_mask(span_start_idx, span_end_idx, passage_length)
        return prob_p

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        rouge_l = self._rouge_metrics.get_metric(reset)
        bleu_4 = self._bleu_metrics.get_metric(reset)
        return {'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'bleu_4': bleu_4,
                'rouge_L': rouge_l}
