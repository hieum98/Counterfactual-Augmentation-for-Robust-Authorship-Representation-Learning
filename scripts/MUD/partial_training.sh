python ERLAS/main.py --dataset_name MUD --token_max_length 128 --index_by_BM25 --index_by_dense_retriever --BM25_percentage 0.5 --dense_percentage 0.5 --gpus 4 --use_gc --gc_minibatch_size 16 --learning_rate 2e-5 --learning_rate_scaling --num_epoch 1 --do_learn --experiment_id MUD --version data_0.1 --training_percentage 0.1

# python ERLAS/main.py --dataset_name MUD --token_max_length 128 --index_by_BM25 --index_by_dense_retriever --BM25_percentage 0.5 --dense_percentage 0.5 --gpus 3 --use_gc --gc_minibatch_size 16 --learning_rate 2e-5 --learning_rate_scaling --num_epoch 1 --do_learn --experiment_id MUD --version data_0.25 --training_percentage 0.25

python ERLAS/main.py --dataset_name MUD --token_max_length 128 --index_by_BM25 --index_by_dense_retriever --BM25_percentage 0.5 --dense_percentage 0.5 --gpus 4 --use_gc --gc_minibatch_size 16 --learning_rate 2e-5 --learning_rate_scaling --num_epoch 1 --do_learn --experiment_id MUD --version data_0.5 --training_percentage 0.5

