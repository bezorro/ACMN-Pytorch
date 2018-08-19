from __future__ import print_function, division
from tqdm import tqdm
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from vqa_lab.evaluator import LossEvaluator, AccuracyEvaluator, VQAV2ScoreEvaluator
from vqa_lab.display import SingleNumberVizer
from vqa_lab.utils import print_save

import argparse
parser = argparse.ArgumentParser(description='PyTorch CapsNet-Finetune(Openended)')
# training settings
parser.add_argument('--batch_size', type=int, default=32, help="training batch size")
parser.add_argument('--run_dataset', type=str, default='clevr', choices=['clevr'], help='training datatset')
parser.add_argument('--train_dataset', type=str, default='train', choices=['train', 'train+val', 'val'], help='training datatset')
parser.add_argument('--run_model', type=str, default='restree', choices=['restree', 'rbn'], help='training model')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--max_epoch', type=int, default=1000, help='max epoches')
parser.add_argument('--seed', type=int, default=99, help='random seed to use. Default=99')
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--no_val', type=bool, default=False)
parser.add_argument('--no_train', type=bool, default=False)
parser.add_argument('--resume', type=str, default=None, help='resume file name')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
# log settings
parser.add_argument('--logdir', type=str, default='logs/train_val', help='dir to tensorboard logs')
opt, _ = parser.parse_known_args()
print_save(opt.logdir, opt)

torch.backends.cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.gpu: torch.cuda.manual_seed(opt.seed)

#------ init settings --------
from shutil import copyfile
copyfile(__file__, os.path.join(opt.logdir, os.path.basename(__file__)))
from tensorboardX import SummaryWriter
writer = SummaryWriter(opt.logdir)
#----------- end -------------

#------ get dataloaders ------
from vqa_lab.data.data_loader import getDateLoader
print('==> Loading datasets :')
Dataloader = getDateLoader(opt.run_dataset)
if opt.train_dataset == 'train' :
	dataset_train, dataset_val  = Dataloader('train', opt), Dataloader('val'  , opt)
	opt.__dict__                = { **opt.__dict__, **dataset_train.dataset.opt }
elif opt.train_dataset == 'train+val' :
	dataset_train = Dataloader('train+val', opt)
	opt.__dict__                = { **opt.__dict__, **dataset_train.dataset.datasets[0].opt }
#----------- end -------------

#------ get mode_lrunner -----
from vqa_lab.model.model_runner import getModelRunner
print('==> Building Network :')
model_runner   = getModelRunner(opt.run_model)(opt)
finished_epoch = model_runner.finished_epoch
#----------- end -------------

#---------- train ------------
def train(epoch):
	if opt.no_train==True: return

	print('==> Train-Epoch {:d} :'.format(epoch))

	LE = LossEvaluator()
	LV = SingleNumberVizer(writer, 'loss', interval = opt.display_interval)

	for i_batch, input_batch in enumerate(tqdm(dataset_train)):

		output_batch = model_runner.train_step(input_batch)
		LE.add_batch(output_batch, input_batch)
		LV.print_result(i_batch + len(dataset_train) * (epoch - 1), output_batch['loss'])
		break
	LE.get_print_result()
#----------- end -------------

#----------- val -------------
def val(epoch):
	if opt.no_val==True: return

	print('==> Evaluate-Epoch {:d} :'.format(epoch))

	LE = LossEvaluator()
	AE = AccuracyEvaluator()
	AV = SingleNumberVizer(writer, 'accuracy')
	LV = SingleNumberVizer(writer, 'test_loss')

	for i_batch, input_batch in enumerate(tqdm(dataset_val)):

		output_batch = model_runner.val_step(input_batch)
		AE.add_batch(output_batch, input_batch)
		LE.add_batch(output_batch, input_batch)
		break
	
	test_loss = LE.get_print_result()
	acc       = AE.get_print_result()
	AV.print_result(epoch, acc)
	LV.print_result(epoch, test_loss)
#----------- end -------------

#----------- main ------------
for epoch in range(finished_epoch + 1, opt.max_epoch + 1):

	train(epoch)
	model_runner.save_checkpoint(epoch, opt.logdir)
	val(epoch)
#----------- end -------------

