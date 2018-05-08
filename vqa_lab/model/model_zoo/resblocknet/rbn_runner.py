import argparse
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.autograd import Variable

from vqa_lab.model.model_runner import ModelRunner
from vqa_lab.utils import print_save, forced_copydir

def ModelRunner_RBN(opt):

	parser = argparse.ArgumentParser(description='PyTorch RBNRunner')
	parser.add_argument('--rbn_vocab_size', type=int, default=None)
	parser.add_argument('--rbn_out_size', type=int, default=None)
	parser.add_argument('--rbn_word_emb', type=int, default=300)
	parser.add_argument('--rbn_lr', type=float, default=3e-4)
	parser.add_argument('--rbn_beta1', type=float, default=0.9)
	parser.add_argument('--rbn_beta2', type=float, default=0.999)
	parser.add_argument('--rbn_weight_decay', type=float, default=1e-5)
	parser.add_argument('--rbn_gpu', type=bool, default=False)
	parser.add_argument('--rbn_resume', type=str, default=None)
	parser.add_argument('--rbn_logdir', type=str, default=None)
	my_opt, _ = parser.parse_known_args()
	prefix_len = len('rbn_')

	from .resblocknet import ResBlockNet
	my_opt.rbn_vocab_size = opt.vocab_size
	my_opt.rbn_out_size   = opt.out_vocab_size
	my_opt.rbn_gpu 	      = opt.gpu
	my_opt.rbn_logdir     = opt.logdir

	print_save(my_opt.rbn_logdir, my_opt)
	forced_copydir(os.path.dirname(__file__), os.path.join(my_opt.rbn_logdir, os.path.basename(os.path.dirname(__file__))))

	def forward(model, input, volatile = False, only_forward = False):

		questions, images = Variable(input['question'], volatile=volatile), Variable(input['image'], volatile=volatile)
		
		if my_opt.rbn_gpu :
			
			questions, images = questions.cuda(), images.cuda()

		predicts = model(questions, images)

		output = { 'predicts': predicts.data.cpu() }

		if only_forward == False : 
			
			answers = Variable(input['answer'], volatile=volatile)

			if my_opt.rbn_gpu : answers = answers.cuda()

			output['loss'] = F.cross_entropy(predicts, answers)

		return output
	
	model_opt = argparse.Namespace(**{ k[prefix_len:] : v for k, v in my_opt.__dict__.items() }) # remove opt prefix

	return ModelRunner(model=ResBlockNet, model_opt=model_opt, forward_fn=forward, optimizer='adam')