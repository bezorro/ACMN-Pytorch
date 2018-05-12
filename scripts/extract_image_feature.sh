#!/bin/sh

basepath=$(cd `dirname $0`; pwd)
cd $basepath

python extract_res101_hdf5.py --in_path=../data/clevr/CLEVR_v1.0/images/train/ \
                              --out_path=../data/clevr/clevr_res101/features_train.h5

python extract_res101_hdf5.py --in_path=../data/clevr/CLEVR_v1.0/images/val/ \
                              --out_path=../data/clevr/clevr_res101/features_val.h5

python extract_res101_hdf5.py --in_path=../data/clevr/CLEVR_v1.0/images/test/ \
                              --out_path=../data/clevr/clevr_res101/features_test.h5