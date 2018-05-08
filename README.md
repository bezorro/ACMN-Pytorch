# Visual Question Reasoning on General Dependency Tree

This is the code for the paper on CLEVR

 **<a href="https://arxiv.org/abs/1804.00105">Visual Question Reasoning on General Dependency Tree</a>**
 <br>
Qingxing Cao,
 <a href='https://www.cs.cmu.edu/~xiaodan1/'>Xiaodan Liang</a>,
 <a href='https://bezorro.github.io/'>Bailin Li</a>,
 <a href='https://sites.google.com/site/ligb86/home/'>Guanbin Li</a>,
 <a href='http://www.linliang.net/'>Liang Lin</a>,
 <br>
 Presented at [CVPR 2018](http://cvpr2018.thecvf.com/) (Spotlight Presentation)

 <div align="center">
  <img src="https://github.com/bezorro/ACMN-Pytorch/blob/master/img/introduction.png" width="450px">
</div>

If you find this code useful in your research then please cite

```
@inproceedings{cao2018acmn,
  title={Visual Question Reasoning on General Dependency Tree},
  author={Qingxing Cao and Xiaodan Liang and Bailing Li and Guanbin Li
          and Liang Lin},
  booktitle={CVPR},
  year={2018}
}
```

## Requirement
  * pytorch 0.3.0.post4
  * python 3.5
  * tensorboardX
  * skimage
  * scipy
  * numpy
  * torchvision
  * h5py
  * tqdm

## Data Preprocessing
Before you can train any models, you need to download the datasets; you also need to preprocess questions, and extract features for the images.

### Step 1: Download the data
You can follow the [instructions](https://github.com/jcjohnson/clevr-iep/blob/master/TRAINING.md) to download CLEVR dataset. Or you can skip this step by using our preprossed questions data(Step 2) and extracted image features(Step 3).

### Step 2: Preprocess Questions
Codes for preprocessing would be available soon. For now you can download our preprocessed data with the following commands:
```sh
$ sh ./data/clevr/download_preprocessed_questions.sh
```

### Step 3: Extract Image Features
Codes for image feature extraction and our extracted image features would be available soon. For now you can follow the [instructions](https://github.com/jcjohnson/clevr-iep/blob/master/TRAINING.md) to extract image features.
We assume the extracted features `features_train.h5`, `features_val.h5`, `features_test.h5` are placed in `./data/clevr/clevr_res101/`.

## Pretrained Models
You can download the pretrained models with common below. The model will take about 2.6 GB on disk.
```sh
$ sh ./data/clevr/download_pretrained_model.sh
```
It is trained on `CLEVR-train` and can be validate on `CLEVR-val`.


## Training on CLEVR
You can use the `train_val.py` script to train on `CLEVR-train` and validate the model on `CLEVR-val`.
```sh
$ python scripts/train_val.py \
  --train_dataset=train \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
  --resume=data/clevr/clevr_pretrained_model.pth
```
The below script has the hyperparameters and settings to reproduce ACMN CLEVR results.
```
$ sh scripts/train_val.sh
```

## Evaluation
You can use `train_val.py` to simply evaluate the model on `CLEVR-val`.
```sh
$ python scripts/train_val.py \
  --no_train=True \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
  --resume=data/clevr/clevr_pretrained_model.pth
```
If you want to train on `CLEVR-train+val` and evaluate on `CLEVR-test`.
Firstly, use `train_val.py` with `--no_val` option to train.
```
$ python scripts/train_val.py \
  --no_val=True \
  --train_dataset=train+val \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
```
Then use `test.py` to generate `CLEVR-test` results in `.json` format.
```
$ python scripts/test.py \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
  --resume=data/clevr/clevr_pretrained_model.pth
```

## TODO
```
1. data preprocess code
2. upload pytorch image feature extract code
3. visualize attention map
```