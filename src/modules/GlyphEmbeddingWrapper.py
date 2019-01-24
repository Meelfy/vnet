import torch
import sys

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

try:
    sys.path.append("/home/meelfy/working/WordLanguageModel/")
    from glyph_embedding.models.char_glyph_embedding import CharGlyphEmbedding
    from glyph_embedding.utils.default_config import GlyphEmbeddingConfig
except ModuleNotFoundError as e:
    raise e


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
                 glyph_config,
                 encoder) -> None:
        super(GlyphEmbeddingWrapper, self).__init__()
        self.glyph_config = glyph_config
        self.glyph_config.idx2char = vocab._index_to_token['token_characters']
        self.glyph_embedding = CharGlyphEmbedding(self.glyph_config)
        self._encoder = TimeDistributed(encoder)
        self.using_glyph = True

    def get_output_dim(self) -> int:
        return self._encoder._module.get_output_dim()

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        mask = (token_characters != 0).long()
        character_embedding, loss = self.glyph_embedding(token_characters)
        return self._encoder(character_embedding, mask), loss

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'GlyphEmbeddingWrapper':
        # glyph_config
        glyph_config = GlyphEmbeddingConfig()
        glyph_config.output_size = params.pop_int("output_size", 300)
        glyph_config.use_highway = True
        glyph_config.dropout = params.pop_float("dropout", 0.0)
        glyph_config.glyph_embsize = params.pop_int("glyph_embsize", 256)
        glyph_config.use_batch_norm = params.pop_bool("use_batch_norm", False)
        # encoder_config
        encoder_params: Params = params.pop("encoder")
        encoder = Seq2VecEncoder.from_params(encoder_params)
        params.assert_empty(cls.__name__)
        return cls(vocab, glyph_config, encoder)
