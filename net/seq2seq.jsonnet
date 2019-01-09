{
  "vocabulary":{
        "pretrained_files":{
            "tokens": "/home/meefly/data/msmarco/vocabulary/tokens.txt",
            "token_characters": "/home/meefly/data/msmarco/vocabulary/token_characters.txt",
            "non_padded_namespaces": "/home/meefly/data/msmarco/vocabulary/non_padded_namespaces.txt"
        },
        "only_include_pretrained_words":true
    },
  "dataset_reader": {
    "type": "seq2seq",
    "source_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "spacy"
      }
    },
    "target_tokenizer": {
      "type": "word"
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "namespace": "target_tokens"
      }
    }
  },
  "train_data_path": "/home/meefly/data/vnet_seq2seq/train.tsv",
  "validation_data_path": "/home/meefly/data/vnet_seq2seq/dev.tsv",
  "model": {
    "type": "simple_seq2seq",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 25,
          "trainable": true
        },
        "pos_tags": {
          "type": "embedding",
          "vocab_namespace": "pos",
          "embedding_dim": 0
        },
        "ner_tags": {
          "type": "embedding",
          "vocab_namespace": "ner",
          "embedding_dim": 0
        },
        "dependency_label": {
          "type": "embedding",
          "vocab_namespace": "dependencies",
          "embedding_dim": 0
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 25,
      "hidden_size": 10,
      "num_layers": 1
    },
    "max_decoding_steps": 40,
    "target_embedding_dim": 30,
    "target_namespace": "target_tokens",
    "attention": {
      "type": "dot_product"
    },
    "beam_size": 5
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 80,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 10,
    "cuda_device": 3,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}