 {
    "dataset_reader":{
        "type":"dureader_multi_passage_limited",
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
        "lazy": true,
        "max_samples": 1000
    },
    "vocabulary":{
        "directory_path":"/data/nfsdata/meijie/data/dureader/vocabulary/",
    },
    "train_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/trainset/train.json",
    // "validation_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/trainset/train.json",
    "validation_data_path":"/data/nfsdata/meijie/data/dureader/preprocessed/devset/dev.json",
    "model":{
        "type":"vnet",
        // "type":"bidaf_zh",
        "text_field_embedder":{
            "type": "basic_with_loss",
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "embedding_dim":300,
                    "trainable":true
                }
                ,
                // "token_characters":{
                //     "type":"glyph_encoder",
                //     "glyph_embsize": 128,
                //     "output_size": 128,
                //     "use_batch_norm": true,
                //     "font_channels": 8,
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
        "loss_ratio": 0.1,
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
        "batch_size": 32
    },
    // "iterator":{
    //     "type":"multiprocess",
    //     "base_iterator":{
    //         "type":"bucket",
    //         "sorting_keys":[["question", "num_tokens"]],
    //         "biggest_batch_first":true,
    //         "batch_size": 2
    //     },
    //     "num_workers": 2
    // },
    "trainer":{
        "num_epochs":8,
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
            "lr": 0.0001
        }
    }
}
