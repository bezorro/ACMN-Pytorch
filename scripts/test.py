from tqdm import tqdm
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from vqa_lab.evaluator import AccuracyEvaluator

parser = argparse.ArgumentParser(description='PyTorch resTree test ON CLEVR')
parser.add_argument('--batch_size', type=int, default=64, help="training batch size")
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu?')
parser.add_argument('--run_model', type=str, default='restree', choices=['restree', 'rbn'], help='run model')
parser.add_argument('--run_dataset', type=str, default='clevr', choices=['clevr'])
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'])
parser.add_argument('--resume', type=str, default=None, help='resume file name')
parser.add_argument('--logdir', type=str, default='logs/test', help='dir to tensorboard logs')
opt, _ = parser.parse_known_args()

torch.backends.cudnn.benchmark = True

#------ get dataloaders ------
from vqa_lab.data.data_loader import getDateLoader
print('==> Loading datasets :')
Dataloader   = getDateLoader(opt.run_dataset)
dataset_test = Dataloader(opt.eval_type, opt)
opt.__dict__ = { **opt.__dict__, **dataset_test.dataset.opt }
#----------- end -------------

#------ get mode_lrunner -----
from vqa_lab.model.model_runner import getModelRunner
print('==> Building Network :')
model_runner = getModelRunner(opt.run_model)(opt)
#----------- end -------------

#----------- main ------------
print('Generating test results...')
answer_map = {}

for i_batch, input_batch in enumerate(tqdm(dataset_test)):

    output_batch = model_runner.test_step(input_batch)
    pred = output_batch['predicts'].max(1)[1]

    for i in range(pred.size(0)):
        answer_map[input_batch['qid'][i]] = dataset_test.dataset.ansVocabRev[pred[i]]

with open(os.path.join(opt.logdir, 'CLEVR_TEST_results.txt' ), 'w') as f:
    for i in range(len(answer_map.keys())):
        print(answer_map[i], file = f)
    print(os.path.join(opt.logdir, 'CLEVR_TEST_results.txt' ) + ' saved.')
#----------- end -------------
