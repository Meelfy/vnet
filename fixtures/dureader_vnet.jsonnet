{
    "dataset_reader":{
        "type":"msmarco_multi_passage_limited",
        "token_indexers":{
            "tokens":{
                "type":"single_id",
                "lowercase_tokens":true
            }
            ,
            "token_characters":{
                "type":"characters"
            }
        },
        "lazy":true,
        "language": "zh",
        "passage_length_limit":500,
        "question_length_limit":50
    },
    // "train_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/trainset/test.txt",
    // "validation_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/trainset/test.txt",
    // "train_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/devset/zhidao.dev.json",
    // "validation_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/devset/search.dev.json",
    "train_data_path":"/home/meefly/working/vnet/fixtures/big_samples_dureader.json",
    "validation_data_path":"/home/meefly/working/vnet/fixtures/big_samples_dureader.json",
    "model":{
        "type":"vnet",
        "text_field_embedder":{
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "pretrained_file":"/data/nfsdata/nlp/embeddings/chinese/word_embedding300.data",
                    "embedding_dim":300,
                    "trainable":true
                }
                ,
                "token_characters":{
                    "type":"glyph_encoder",
                    "output_size": 100,
                    "glyph_embsize": 256,
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
        "modeling_layer":{
            "type":"lstm",
            "bidirectional":true,
            "input_size":800,
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
        "max_num_passages": 5,
        "language": "zh",
        "dropout":0.2
    },
    "iterator":{
        "type":"bucket",
        "sorting_keys":[["question", "num_tokens"]],
        "biggest_batch_first":true,
        "batch_size":1
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
