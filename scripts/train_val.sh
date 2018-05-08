#!/bin/sh

basepath=$(cd `dirname $0`; pwd)
cd $basepath

python train.py \
	--tree_lr=9e-5 \
	--tree_beta1=0.9 \
	--tree_beta2=0.999 \
	--tree_weight_decay=0 \
	--tree_optim='adam' \
	--tree_vocab_size=81 \
	--tree_out_vocab_size=29 \
	--tree_word_emb=300 \
	--tree_commom_emb=256 \
	--tree_dropout=0.0 \
	--tree_encode='LSTM' \
	--tree_img_emb=128 \
	--tree_sent_len=45 \
	--tree_sentence_emb=2048 \
	--logdir=logs/train_val \
	--max_epoch=7 \
	--batch_size=32 \
	--seed=99 \
	--run_model=restree \
	--run_dataset=clevr \
	--clevr_image_source=h5 \
	--clevr_load_trees=True \
	--clevr_qa_dir=../data/clevr/clevr_qa_dir/ \
	--clevr_img_h5=../data/clevr/clevr_res101/

python train.py \
	--tree_lr=5e-6 \
	--tree_beta1=0.9 \
	--tree_beta2=0.999 \
	--tree_weight_decay=0 \
	--tree_optim='adam' \
	--tree_vocab_size=81 \
	--tree_out_vocab_size=29 \
	--tree_word_emb=300 \
	--tree_commom_emb=256 \
	--tree_dropout=0.0 \
	--tree_encode='LSTM' \
	--tree_img_emb=128 \
	--tree_sent_len=45 \
	--tree_sentence_emb=2048 \
	--logdir=logs/train_val \
	--max_epoch=9 \
	--batch_size=32 \
	--seed=99 \
	--run_model=restree \
	--run_dataset=clevr \
	--clevr_image_source=h5 \
	--clevr_load_trees=True \
	--logdir=logs/train_val \
	--resume=logs/train_val/model_epoch_7.pth \
	--clevr_qa_dir=../data/clevr/clevr_qa_dir/ \
	--clevr_img_h5=../data/clevr/clevr_res101/
