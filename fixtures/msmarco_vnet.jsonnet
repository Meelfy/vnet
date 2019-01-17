{
    "vocabulary":{
        "pretrained_files":{
            "tokens": "/home/meefly/data/msmarco/vocabulary/tokens.txt",
            "token_characters": "/home/meefly/data/msmarco/vocabulary/token_characters.txt",
            "non_padded_namespaces": "/home/meefly/data/msmarco/vocabulary/non_padded_namespaces.txt"
        },
        "only_include_pretrained_words":true
    },
    "dataset_reader":{
        "type":"msmarco_multi_passage_limited",
        "token_indexers":{
            "tokens":{
                "type":"single_id",
                "lowercase_tokens":true
            },
            "token_characters":{
                "type":"characters"
            }
        },
        "lazy":true,
        "passage_length_limit":400,
        "question_length_limit":50
    },
    "train_data_path":"/home/meefly/misc/train.json",
    "validation_data_path":"/home/meefly/misc/train.json",
    // "validation_data_path":"/home/meefly/misc/dev.json",
    "model":{
        "type":"vnet",
        "text_field_embedder":{
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "pretrained_file":"/data/nfsdata/meijie/data/WordEmb/glove.6B.300d.txt",
                    "embedding_dim":300,
                    "trainable":false
                },
                "token_characters":{
                    "type":"character_encoding",
                    "embedding":{
                        "num_embeddings":9000,
                        "embedding_dim":16
                    },
                    "encoder":{
                        "type":"cnn",
                        "embedding_dim":16,
                        "num_filters":100,
                        "ngram_filter_sizes":[
                            5
                        ]
                    },
                    "dropout":0.2
                }
            }
        },
        "num_highway_layers":2,
        "phrase_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":400,
            "hidden_size":100,
            "num_layers":2,
            "dropout":0.2
        },
        "match_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":1000,
            "hidden_size":500,
            "num_layers":2,
            "dropout":0.2
        },
        "modeling_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":800,
            "hidden_size":100,
            "num_layers":2,
            "dropout":0.2
        },
        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 400,
            "tensor_2_dim": 400,
            "combination": "x,y,x*y"
        },
        "pointer_net": {
            "bidirectional": false,
            "input_size": 1000,
            "hidden_dim": 200,
            "lstm_layers": 2,
            "dropout": 0.0
        },
        "span_end_lstm":{
            "type":"lstm",
            "bidirectional":false,
            "input_size":1000,
            "hidden_size":200,
            "num_layers":2,
            "dropout":0.2
        },
        "ptr_dim":200,
        "max_num_passages": 10,
        "language": "en",
        "dropout":0.2
    },
    "iterator":{
        "type":"bucket",
        "sorting_keys":[["question", "num_tokens"]],
        "batch_size":2
    },
    "trainer":{
        "num_epochs":10,
        "grad_norm":5,
        "patience":10,
        "validation_metric":"+rouge_L",
        "cuda_device":3,
        "learning_rate_scheduler":{
            "type":"reduce_on_plateau",
            "factor":0.5,
            "mode":"max",
            "patience":4
        },
        "optimizer":{
            "type":"adam",
            "betas":[
                0.9,
                0.9
            ],
            "lr": 0.0004
        }
    }
}
