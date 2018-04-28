#!/bin/sh

basepath=$(cd `dirname $0`; pwd)
cd $basepath
mkdir clevr_qa_dir
cd clevr_qa_dir
wget --max-redirect=20 -O clevr_qa_dir.zip https://www.dropbox.com/sh/lutwyt2p8vd5by1/AABneKTyo50Gpc2xJRyOFYVsa?dl=1
unzip clevr_qa_dir.zip
rm clevr_qa_dir.zip