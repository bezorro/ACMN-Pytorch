import torch
import argparse
from vqa_lab.utils import print_save, update_opt_remove_prefix
from torch.utils.data import DataLoader

def DataloaderClevr(mode, opt = {}, dl = False, opt_prefix = 'clevr_'):

	from vqa_lab.config import CLEVR_QA_DIR, CLEVR_IMG_H5, CLEVR_IMG_PNG
	parser = argparse.ArgumentParser(description='PyTorch CLEVR DataLoaer')
	parser.add_argument('--clevr_image_source', type=str, default='h5', help='h5|png')
	parser.add_argument('--clevr_qa_dir', type=str, default=CLEVR_QA_DIR)
	parser.add_argument('--clevr_img_h5', type=str, default=CLEVR_IMG_H5)
	parser.add_argument('--clevr_img_png', type=str, default=CLEVR_IMG_PNG)
	parser.add_argument('--clevr_load_trees', type=bool, default=True)
	parser.add_argument('--clevr_load_png', type=bool, default=False)
	my_opt, _  = parser.parse_known_args()
	prefix_len = len(opt_prefix)

	if isinstance(opt, dict) : opt = argparse.Namespace(opt)

	my_opt = update_opt_remove_prefix(my_opt, opt, opt_prefix) # my_opt = { opt_prefix + '...' }

	if 'logdir' in opt.__dict__: print_save(opt.logdir, my_opt, to_screen=False)

	from .dataset_clevr import clevrDataset

	def my_collate(batch):

		raw_samples = { key: [d[key] for d in batch] for key in batch[0] }

		samples = {
                    'question': torch.stack(raw_samples['question'])   ,
                    'image'   : torch.stack(raw_samples['image'])      ,
                    'qid'     : torch.LongTensor(raw_samples['qid'])   ,
                    'answer'  : torch.LongTensor(raw_samples['answer']),
                }

		if 'tree'       in raw_samples : 
			samples['tree'] = raw_samples['tree']
		if 'img_png'    in raw_samples :
			samples['img_png']    = torch.stack(raw_samples['img_png'])

		return samples

	dataset_opt = argparse.Namespace(**{ k[prefix_len:] : v for k, v in my_opt.__dict__.items() }) # remove opt prefix

	dataset = clevrDataset(dataset_opt, mode)

	return DataLoader(dataset                      ,
                    batch_size  = opt.batch_size   ,
                    collate_fn  = my_collate       ,
                    num_workers = opt.threads  	   ,
                    shuffle     = (mode == 'train'),
                    drop_last   = dl)

def DataloaderSclevr(mode, opt = {}, dl = False, opt_prefix = 'sclevr_'):

	from vqa_lab.config import SCLEVR_QA_DIR
	parser = argparse.ArgumentParser(description='PyTorch Sort-of-CLEVR DataLoader(Openended)')
	parser.add_argument('--sclevr_qa_dir', type=str, default=SCLEVR_QA_DIR)
	my_opt, _  = parser.parse_known_args()
	prefix_len = len(opt_prefix)

	if isinstance(opt, dict) : opt = argparse.Namespace(**opt)

	my_opt = update_opt_remove_prefix(my_opt, opt, opt_prefix)

	if 'logdir' in opt.__dict__: print_save(opt.logdir, my_opt, to_screen=False)

	def my_collate(batch):

		raw_samples = { key: [d[key] for d in batch] for key in batch[0] }
        
		return {    
                    'question': torch.stack(raw_samples['question'])   ,
                    'image'   : torch.stack(raw_samples['image'])      ,
                    'qid'     : torch.LongTensor(raw_samples['qid'])   ,
                    'answer'  : torch.LongTensor(raw_samples['answer']),
               }

	dataset_opt = argparse.Namespace(**{ k[prefix_len:] : v for k, v in my_opt.__dict__.items() }) # remove opt prefix
          
	from .dataset_sclevr import sclevrDataset
	dataset = sclevrDataset(dataset_opt, mode)

	return DataLoader(dataset     = dataset          ,
                      batch_size  = opt.batch_size   ,
                      collate_fn  = my_collate       ,
                      num_workers = opt.threads      ,
                      shuffle     = (mode == 'train'),
                      drop_last   = dl)

def DataloaderVQAv2(mode, opt = {}, dl = False, opt_prefix = 'vqav2_'):

	from vqa_lab.config import VQAV2_QA_DIR, VQAV2_IMG_H5, VQAV2_IMG_BU, VQAV2_IMG_JPG
	parser = argparse.ArgumentParser(description='PyTorch VQAv2 DataLoader(Openended)')
	parser.add_argument('--vqav2_image_source', type=str, default='BU', help='h5|jpg|BU')
	parser.add_argument('--vqav2_img_h5', type=str, default=VQAV2_IMG_H5)
	parser.add_argument('--vqav2_img_BU', type=str, default=VQAV2_IMG_BU)
	parser.add_argument('--vqav2_img_jpg', type=str, default=VQAV2_IMG_JPG)
	parser.add_argument('--vqav2_qa_dir', type=str, default=VQAV2_QA_DIR)
	parser.add_argument('--vqav2_load_trees', type=bool, default=True)
	parser.add_argument('--vqav2_load_jpg', type=bool, default=False)
	my_opt, _  = parser.parse_known_args()
	prefix_len = len(opt_prefix)

	if isinstance(opt, dict) : opt = argparse.Namespace(**opt)

	my_opt = update_opt_remove_prefix(my_opt, opt, opt_prefix)

	if 'logdir' in opt.__dict__: print_save(opt.logdir, my_opt, to_screen=False)

	def my_collate(batch):

		raw_samples = { key: [d[key] for d in batch] for key in batch[0] }

		samples = {    
                    'question'   : torch.stack(raw_samples['question'])   ,
                    'image'      : torch.stack(raw_samples['image'])      ,
                    'qid'        : torch.LongTensor(raw_samples['qid'])   ,
                  }

		if 'answer'     in raw_samples : 
			samples['answer']     = torch.LongTensor(raw_samples['answer'])
		if 'raw_answer' in raw_samples : 
			samples['raw_answer'] = torch.stack(raw_samples['raw_answer'])
		if 'tree'       in raw_samples : 
			samples['tree']       = raw_samples['tree']
		if 'img_jpg'    in raw_samples :
			samples['img_jpg']    = torch.stack(raw_samples['img_jpg'])

		return samples

	dataset_opt = argparse.Namespace(**{ k[prefix_len:] : v for k, v in my_opt.__dict__.items() }) # remove opt prefix
          
	from .dataset_vqav2 import vqa2Dataset
	from functools import reduce
	dataset = reduce(lambda x, y : x + y, [vqa2Dataset(dataset_opt, m) for m in mode.split('+')])

	return DataLoader(dataset     = dataset          ,
                      batch_size  = opt.batch_size   ,
                      collate_fn  = my_collate       ,
                      num_workers = opt.threads      ,
                      shuffle     = ('train' in mode),
                      drop_last   = dl)

def DataloaderFigureQA(mode, opt = {}, dl = False, opt_prefix = 'figureqa_'):

	from vqa_lab.config import FIGUREQA_IMG_PNG, FIGUREQA_QA_DIR
	parser = argparse.ArgumentParser(description='PyTorch FigureQA DataLoader(Openended)')
	parser.add_argument('--figureqa_img_png', type=str, default=FIGUREQA_IMG_PNG)
	parser.add_argument('--figureqa_qa_dir', type=str, default=FIGUREQA_QA_DIR)
	parser.add_argument('--figureqa_load_trees', type=bool, default=True)
	my_opt, _  = parser.parse_known_args()
	prefix_len = len(opt_prefix)

	if isinstance(opt, dict) : opt = argparse.Namespace(**opt)

	my_opt = update_opt_remove_prefix(my_opt, opt, opt_prefix)

	if 'logdir' in opt.__dict__: print_save(opt.logdir, my_opt, to_screen=False)

	def my_collate(batch):

		raw_samples = { key: [d[key] for d in batch] for key in batch[0] }

		samples = {    
                    'question'   : torch.stack(raw_samples['question'])   ,
                    'image'      : torch.stack(raw_samples['image'])      ,
                    'qid'        : torch.LongTensor(raw_samples['qid'])   ,
                  }

		if 'answer'     in raw_samples : 
			samples['answer']     = torch.LongTensor(raw_samples['answer'])
		if 'tree'       in raw_samples : 
			samples['tree']       = raw_samples['tree']

		return samples

	dataset_opt = argparse.Namespace(**{ k[prefix_len:] : v for k, v in my_opt.__dict__.items() }) # remove opt prefix
          
	from .dataset_figureqa import figureqaDataset
	from functools import reduce
	dataset = reduce(lambda x, y : x + y, [figureqaDataset(dataset_opt, m) for m in mode.split('+')])

	return DataLoader(dataset     = dataset          ,
                      batch_size  = opt.batch_size   ,
                      collate_fn  = my_collate       ,
                      num_workers = opt.threads      ,
                      shuffle     = ('train' in mode),
                      drop_last   = dl)


def getDateLoader(dataset):

	return	{
				'sclevr'   : DataloaderSclevr   ,
				'clevr'    : DataloaderClevr    ,
				'vqav2'    : DataloaderVQAv2    ,
				'figureqa' : DataloaderFigureQA ,
			}[dataset]
