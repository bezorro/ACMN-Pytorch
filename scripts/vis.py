from __future__ import print_function, division
from tqdm import tqdm
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from vqa_lab.display import AttTreeImgVizer
from vqa_lab.utils import print_save

import argparse
parser = argparse.ArgumentParser(description='PyTorch resTree-Visualize(Openended)')
# training settings
parser.add_argument('--batch_size', type=int, default=4, help="training batch size")
parser.add_argument('--run_dataset', type=str, default='clevr', choices=['clevr'], help='training datatset')
parser.add_argument('--run_model', type=str, default='restree', choices=['restree'], help='training model')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=66, help='random seed to use. Default=123')
parser.add_argument('--resume', type=str, default='../data/clevr/clevr_pretrained_model.pth', help='resume file name')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
# log settings
parser.add_argument('--logdir', type=str, default='logs/test', help='dir to tensorboard logs')
opt, _ = parser.parse_known_args()
print_save(opt.logdir, opt)

torch.manual_seed(opt.seed)
if opt.gpu: torch.cuda.manual_seed(opt.seed)

#------ get dataloaders ------
from vqa_lab.data.data_loader import getDateLoader
print('==> Loading datasets :')
Dataloader   = getDateLoader(opt.run_dataset)
dataset_run  = Dataloader('train', opt)
opt.__dict__ = { **opt.__dict__, **dataset_run.dataset.opt }
#----------- end -------------

#------ get mode_lrunner -----
from vqa_lab.model.model_runner import getModelRunner
print('==> Building Network :')
model_runner   = getModelRunner(opt.run_model)(opt)
#----------- end -------------

#----------- main ------------
AV = AttTreeImgVizer(opt.logdir, opt.batch_size, dataset_run.dataset.VocabRev, dataset_run.dataset.ansVocabRev)
for i_batch, input_batch in enumerate(tqdm(dataset_run)):

	output_batch = model_runner.test_step(input_batch)

	AV.print_result(images=input_batch['img_png'],
					node_values=output_batch['node_values'][1],
					questions=input_batch['question'],
					trees=input_batch['tree'],
					predicts=output_batch['predicts'],
					answers=input_batch['answer']) 
#----------- end -------------