import torch
import torch.optim as optim

from vqa_lab.utils import print_save

class ModelRunner(object):

	def __init__(self, model, model_opt, forward_fn, optimizer, lr_scheduler = None):
		super(ModelRunner, self).__init__()

		self.model   	  = model(model_opt)
		self.optimizer    = optimizer
		self.lr_scheduler = lr_scheduler

		if self.optimizer == 'adam' : 
			self.optimizer = optim.Adam(self.model.parameters() 				, \
										lr=model_opt.lr                         , \
										betas=(model_opt.beta1, model_opt.beta2), \
										weight_decay=model_opt.weight_decay)
		elif self.optimizer == 'sgd':
			self.optimizer = optim.SGD(self.model.parameters() 				, \
										lr=model_opt.lr                     , \
										momentum = model_opt.momentum       , \
										weight_decay=model_opt.weight_decay)
		elif self.optimizer == 'adamax':
			self.optimizer = optim.Adamax(self.model.parameters(), lr = model_opt.lr)

		if lr_scheduler == 'step' : self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, model_opt.lr_step_size, model_opt.lr_gamma)

		self.gpu 	 	= model_opt.gpu
		self.forward_fn = forward_fn

		self.finished_epoch = 0

		device = torch.device('cuda' if self.gpu else "cpu")
		self.model = self.model.to(device)

		if model_opt.resume is not None:
			self.set_model_weights(model_opt.resume)
		else:
			self.set_model_weights('kaiming')

		print_save(model_opt.logdir, self.model)

	def set_model_weights(self, init_mothod = 'kaiming'):

		if init_mothod == 'kaiming':

			def init_weights(m):

				if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)): torch.nn.init.kaiming_uniform_(m.weight)

			self.model.apply(init_weights)

		else:

			print('resume from ' + init_mothod)
			checkpoint     		= torch.load(init_mothod)
			self.finished_epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['net'])

	def train_step(self, sample):

		self.model.train()

		output = self.forward_fn(self.model, sample, volatile=False, only_forward=False)

		self.optimizer.zero_grad()
		output['loss'].backward()
		output['loss'] = output['loss'].item()
		self.optimizer.step()
		if self.lr_scheduler is not None : self.lr_scheduler.step()

		return output

	def val_step(self, sample):

		self.model.eval()
		
		output = self.forward_fn(self.model, sample, volatile=True, only_forward=False)

		output['loss'] = output['loss'].item()

		return output

	def test_step(self, sample):

		self.model.eval()

		output = self.forward_fn(self.model, sample, volatile=True, only_forward=True)

		return output

	def save_checkpoint(self, epoch, logdir):

		model_out_path	    = logdir + '/model_epoch_{}.pth'.format(epoch)
		self.finished_epoch = epoch
		checkpoint     		= { 'net' : self.model.state_dict(), 'optimizer' : self.optimizer.state_dict(), 'epoch' : epoch }
		torch.save(checkpoint, model_out_path)
		print('Checkpoint saved to {}'.format(model_out_path))

#---------------------------- get model runner ---------------------------
from vqa_lab.model.model_zoo.resblocknet.rbn_runner import ModelRunner_RBN
from vqa_lab.model.model_zoo.restree.tree_runner    import ModelRunner_Tree

def getModelRunner(model):

	return	{
				'rbn' :     ModelRunner_RBN , \
				'restree' : ModelRunner_Tree, \
			}[model]
#----------------------------------- end ---------------------------------