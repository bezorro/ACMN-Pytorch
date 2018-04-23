#-------------------------------------- QA file configuration ------------------------------------------
#-- what data to use for training
TRAIN_DATA_SPLITS = 'genome_train'

#-- what data to use for the vocabulary
QUESTION_VOCAB_SPACE = 'train+val+v7w_train'#'train+val+genome/genome_train+v7w/v7w_train'
ANSWER_VOCAB_SPACE = 'v7w_train'#train/genome/v7w/v7w_train'

#-- vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = '/home/d302/VQA/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = '/home/d302/VQA/PythonEvaluationTools'

#-- location of the data
VQA_PREFIX = '/home/d302/VQA/'
GENOME_PREFIX = '/home/d302/dataset/dataLabel/'

VQA_IM_PREFIX = '/home/wind/Research/DataSets/vqa/images/'
GENOME_IM_PREFIX = '/home/caoqx/datasets/VG_100K/'
V7W_IM_PREFIX = '/home/wind/Research/DataSets/v7w/images/'

DATA_PATHS = {
	'vqa_img_feat': 'data/vqa/features.h5',
	'train': {
		'ques_file': VQA_PREFIX + '/Questions/MultipleChoice_mscoco_train2014_questions.json', #OpenEnded
		'ans_file': VQA_PREFIX + '/Annotations/mscoco_train2014_annotations.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/train2014/COCO_train2014_'
	},
	'val': {
		'ques_file': VQA_PREFIX + '/Questions/MultipleChoice_mscoco_val2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/mscoco_val2014_annotations.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/val2014/COCO_val2014_'
	},
	'test-dev': {
		'ques_file': VQA_PREFIX + '/Questions/MultipleChoice_mscoco_test-dev2015_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/test2015/COCO_test2015_'
	},
	'test': {
		'ques_file': VQA_PREFIX + '/Questions/MultipleChoice_mscoco_test2015_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/test2015/COCO_test2015_'
	},
	#-- TODO it would be nice if genome also followed the same file format as vqa
	'genome_img_feat': 'data/vg/features.h5',
	'genome_train': {
		'genome_file': GENOME_PREFIX + '/question_answers_prepro_aug.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/whole/'
	},
	'genome': {
		'genome_file': GENOME_PREFIX + '/question_answers_prepro_aug.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/whole/'
	},
	'v7w_img_feat': 'data/v7w/features.h5',
	'v7w_train': {
		'v7w_file': GENOME_PREFIX + '/v7w.json',
		'v7w_bbox_file': GENOME_PREFIX + '/v7w_bbox.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/whole/'
	},
	'v7w': {
		'v7w_file': GENOME_PREFIX + '/v7w.json',
		'v7w_bbox_file': GENOME_PREFIX + '/v7w_bbox.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_pool5_bgrms_large/whole/'
	}
}