import os
import torch
import argparse
import torch.nn.functional as F


from vqa_lab.model.model_runner import ModelRunner
from vqa_lab.utils import print_save, forced_copydir

def ModelRunner_Tree(opt):

	parser = argparse.ArgumentParser(description='PyTorch Tree Runner')
	parser.add_argument('--tree_lr', type=float, default=9e-5)
	parser.add_argument('--tree_beta1', type=float, default=0.9)
	parser.add_argument('--tree_beta2', type=float, default=0.999)
	parser.add_argument('--tree_weight_decay', type=float, default=0)
	parser.add_argument('--tree_gpu', type=bool, default=True)
	parser.add_argument('--tree_load_lookup', type=bool, default=False)
	parser.add_argument('--tree_resume', type=str, default=None)
	parser.add_argument('--tree_logdir', type=str, default=None)
	parser.add_argument('--tree_optim', type=str, default='adam')
	parser.add_argument('--tree_lr_scheduler', type=str, default=None)
	parser.add_argument('--tree_lr_step_size', type=int, default=20000)
	parser.add_argument('--tree_lr_gamma', type=float, default=0.7192)

	parser.add_argument('--tree_vocab_size', type=int, default=81)
	parser.add_argument('--tree_out_vocab_size', type=int, default=29)
	parser.add_argument('--tree_word_emb', type=int, default=300)
	parser.add_argument('--tree_commom_emb', type=int, default=256)
	parser.add_argument('--tree_dropout', type=float, default=0.0)
	parser.add_argument('--tree_encode', type=str, default='LSTM')
	parser.add_argument('--tree_img_emb', type=int, default=128)
	parser.add_argument('--tree_sent_len', type=int, default=45)
	parser.add_argument('--tree_sentence_emb', type=int, default=2048)
	
	my_opt, _ = parser.parse_known_args()
	prefix_len = len('tree_')

	from .tree_module import tree_attention_Residual
	my_opt.tree_vocab_size = opt.vocab_size
	my_opt.tree_out_size   = opt.out_vocab_size
	my_opt.tree_gpu 	   = opt.gpu
	my_opt.tree_logdir     = opt.logdir
	my_opt.tree_resume     = opt.resume

	print_save(my_opt.tree_logdir, my_opt)
	forced_copydir(os.path.dirname(__file__), os.path.join(my_opt.tree_logdir, os.path.basename(os.path.dirname(__file__))))

	def forward(model, input, volatile = False, only_forward = False):

		device = torch.device('cuda' if my_opt.tree_gpu else "cpu")
		questions, images, trees = input['question'].to(device), input['image'].to(device), input['tree']

		with torch.set_grad_enabled(not volatile): predicts, node_values = model(questions, images, trees)

		output = { 'predicts': predicts.cpu(), 'node_values': node_values }

		if only_forward == False : 
			
			answers = input['answer'].to(device)

			with torch.set_grad_enabled(not volatile): output['loss'] = F.cross_entropy(predicts, answers)

		return output
	
	model_opt = argparse.Namespace(**{ k[prefix_len:] : v for k, v in my_opt.__dict__.items() }) # remove opt prefix

	return ModelRunner(model=tree_attention_Residual, model_opt=model_opt, forward_fn=forward, optimizer=model_opt.optim, lr_scheduler=model_opt.lr_scheduler)