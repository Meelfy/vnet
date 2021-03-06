{
    "dataset_reader":{
        "type":"msmarco_multi_passage_limited",
        "token_indexers":{
            "tokens":{
                "type":"single_id",
                "lowercase_tokens":true
            },
            "token_characters":{
                "type":"characters",
                "min_padding_length":5
            }
        },
        "lazy": true,
        "max_p_num": 10,
        "max_p_len": 400,
        "max_q_len": 50,
        // "max_samples": 1000,
    },
    "vocabulary":{
        "directory_path":"/data/nfsdata/meijie/data/msmarco/vocabulary/",
    },
    "train_data_path":"/data/nfsdata/meijie/data/msmarco/train.jsonl.instances",
    "validation_data_path":"/data/nfsdata/meijie/data/msmarco/dev.jsonl.instances",
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
                        "num_embeddings":3100,
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
        "max_passage_len": 400,
        "max_num_passages": 10,
        "max_num_character": 5,
        "highway_embedding_size":300,
        "num_highway_layers":1,
        "phrase_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":300,
            "hidden_size":100,
            "num_layers":1,
            // "dropout":0.2
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
            "tensor_1_dim": 300,
            "tensor_2_dim": 300,
            "combination": "x,y,x*y"
        },
        "pointer_net": {
            "bidirectional": false,
            "input_size": 1000,
            "hidden_dim": 200,
            "lstm_layers": 2,
            "dropout": 0.2
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
        "language": "en",
        "dropout":0.2
    },
    "iterator":{
        "type":"bucket",
        "sorting_keys":[["question", "num_tokens"]],
        "batch_size":24
    },
    "trainer":{
        "num_epochs":5,
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
