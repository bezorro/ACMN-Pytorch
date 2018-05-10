#!/bin/sh

basepath=$(cd `dirname $0`; pwd)
cd $basepath
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
rm CLEVR_v1.0.zip