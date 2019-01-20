import torch
import sys

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
try:
    sys.path.append("/home/meefly/working/WordLanguageModel/")
    from glyph_embedding.models.glyph_embedding import GlyphEmbedding
    from glyph_embedding.utils.default_config import GlyphEmbeddingConfig
except ModuleNotFoundError as e:
    print(e)


@TokenEmbedder.register("glyph_encoder")
class GlyphEmbeddingWrapper(TokenEmbedder):
    """
    A ``GlyphEmbeddingWrapper`` takes the output of a
    :class:`~allennlp.data.token_indexers.TokenCharactersIndexer`, which is a tensor of shape
    (batch_size, num_tokens, num_characters), embeds the characters, runs a token-level encoder, and
    returns the result, which is a tensor of shape (batch_size, num_tokens, encoding_dim).  We also
    optionally apply dropout after the token-level encoder.
    We take the embedding and encoding modules as input, so this class is itself quite simple.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 dropout: float = 0.0,
                 output_size: int = 300,
                 glyph_embsize: int = 256) -> None:
        super(GlyphEmbeddingWrapper, self).__init__()
        self.glyph_config = GlyphEmbeddingConfig()
        self.glyph_config.idx2word = vocab._index_to_token['token_characters']
        self.glyph_config.output_size = output_size
        self.glyph_config.use_highway = True
        self.glyph_config.dropout = dropout
        self.glyph_config.glyph_embsize = glyph_embsize
        self.glyph_embedding = GlyphEmbedding(self.glyph_config)
        self._encoder = TimeDistributed(self.glyph_embedding)
        # self._encoder = self.glyph_embedding

    def get_output_dim(self) -> int:
        return self.glyph_config.output_size

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        import pdb
        pdb.set_trace()
        self.glyph_embedding(token_characters)
        emb, loss = self._encoder(token_characters)
        return emb

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'GlyphEmbeddingWrapper':
        # pylint: disable=arguments-differ
        dropout = params.pop_float("dropout", 0.0)
        output_size = params.pop_int("output_size", 300)
        glyph_embsize = params.pop_int("glyph_embsize", 256)
        params.assert_empty(cls.__name__)
        return cls(vocab, dropout, output_size, glyph_embsize)
