{
    "dataset_reader":{
        "type":"msmarco_multi_passage_limited",
        "token_indexers":{
            "tokens":{
                "type":"single_id",
                "lowercase_tokens":true
            }
            // ,
            // "token_characters":{
            //     "type":"characters"
            // }
        },
        // "lazy": true,
        "char_only": true,
        "max_samples": 3,
        "language": "zh",
        "passage_length_limit": 500,
        "question_length_limit": 50
    },
    "train_data_path":"/data/nfsdata/meijie/data/dureader/raw/trainset/train.json.merge_passage",
    "validation_data_path":"/data/nfsdata/meijie/data/dureader/raw/trainset/train.json.merge_passage",
    // "validation_data_path":"/data/nfsdata/meijie/data/dureader/raw/devset/dev.json.merge_passage",
    "model":{
        "type":"vnet",
        "text_field_embedder":{
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    // "pretrained_file":"/data/nfsdata/nlp/embeddings/chinese/word_embedding300.data",
                    // "embedding_dim":300,
                    // "pretrained_file":"/data/nfsdata/nlp/embeddings/chinese/Tencent_AILab_ChineseEmbedding.txt",
                    "embedding_dim":300,
                    "trainable":true
                }
                // ,
                // "token_characters":{
                //     "type":"glyph_encoder",
                //     "glyph_embsize": 128,
                //     "output_size": 128,
                //     "encoder":{
                //         "type":"cnn",
                //         "embedding_dim":128,
                //         "num_filters":100,
                //         "ngram_filter_sizes":[
                //             1
                //         ]
                //     },
                //     "dropout":0.15
                // }
            }
        },
        "highway_embedding_size":300,
        "num_highway_layers":2,
        "phrase_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":300,
            "hidden_size":150,
            "num_layers":2,
            "dropout":0.15
        },
        "modeling_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":1200,
            "hidden_size":150,
            "num_layers":2,
            "dropout":0.15
        },
        "match_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":1500,
            "hidden_size":750,
            "num_layers":2,
            "dropout":0.15
        },
        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 300,
            "tensor_2_dim": 300,
            "combination": "x,y,x*y"
        },
        "pointer_net": {
            "bidirectional": false,
            "input_size": 1500,
            "hidden_dim": 200,
            "lstm_layers": 2,
            "dropout": 0.15
        },
        "span_end_lstm":{
            "type":"lstm",
            "bidirectional":false,
            "input_size":2100,
            "hidden_size":300,
            "num_layers":2,
            "dropout":0.15
        },
        "ptr_dim":200,
        "max_num_passages": 5,
        "max_num_character": 4,
        "language": "zh",
        "dropout":0.15
    },
    "iterator":{
        "type":"bucket",
        "sorting_keys":[["question", "num_tokens"]],
        "biggest_batch_first":true,
        "batch_size": 1
    },
    "trainer":{
        "num_epochs":5,
        "grad_norm":5,
        "patience":10,
        "validation_metric":"+rouge_L",
        "cuda_device":1,
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
            "lr": 0.01
        }
    }
}
