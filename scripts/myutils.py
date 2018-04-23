import json
import torch
import torch.nn.functional as F

# for viz
import numpy as np
import matplotlib.cm as cm
from skimage import io
from skimage import img_as_float
from skimage.transform import resize, rescale
from PIL import Image, ImageDraw
from torch.autograd import Variable
import scipy.misc
import config
import os, textwrap
from glob import glob

def read_json(fname):
    file = open(fname, 'r')
    res = json.load(file)
    file.close()
    return res

def isNotEmpty(graph):
    return graph != []

def IOGT(x1,y1,w1,h1, x2,y2,w2,h2):
    x_int = max(x1, x2)
    y_int = max(y1, y2)
    x2_int = min(x1 + h1, x2 + h2)
    y2_int = min(y1 + w1, y2 + w2)
    interArea = max(0, (x2_int - x_int)) * max(0, (y2_int - y_int))
    gtArea = w1 * h1
    proposalArea = w2 * h2
    return interArea / (gtArea + proposalArea - interArea)

def scene2map(scenes):
    res = torch.zeros(len(scenes), 14, 14)
    for i, scene in enumerate(scenes):
        for obj in scene:
            if 'qa_related' in obj and obj['qa_related'] == True:
                if int(max(1,round(obj['x']*14.))-1) < int(min(14,round((obj['x']+obj['height'])*14.))) \
                and \
                int(max(1,round(obj['y']*14.))-1) < int(min(14,round((obj['y']+obj['width'])*14.))):
                    res[i, \
                    int(max(1,round(obj['x']*14.))-1):int(min(14,round((obj['x']+obj['height'])*14.))), \
                    int(max(1,round(obj['y']*14.))-1):int(min(14,round((obj['y']+obj['width'])*14.)))] = 1
    return res

class vizer(object):
    """docstring for vizer"""
    def __init__(self, opt, writer):
        super(vizer, self).__init__()
        
        self.colormap = cm.ScalarMappable(cmap="jet")
        self.sample_num = min(4, opt.batch_size)
        self.batch_size = opt.batch_size
        self.writer = writer
        self.vdict_rev = read_json(os.path.join(opt.vocab_dir,'VocabRev.json'))
        self.adict_rev = read_json(os.path.join(opt.vocab_dir,'AnsVocabRev.json'))
        self.img_save_prefix = 'logs/running_exp_display/'
        if not os.path.isdir(self.img_save_prefix): os.mkdir(self.img_save_prefix)

    @staticmethod
    def _getImgPath(path):
        # self.qa_dir+'/CLEVR_v1.0/images/'
        if os.path.isfile(config.VQA_IM_PREFIX+'train2014/'+path):
          impath = config.VQA_IM_PREFIX+'train2014/'+path
        elif os.path.isfile(config.VQA_IM_PREFIX+'val2014/'+path):
          impath = config.VQA_IM_PREFIX+'val2014/'+path
        elif os.path.isfile(config.GENOME_IM_PREFIX+'train/'+path):
          impath = config.GENOME_IM_PREFIX+'train/'+path
        elif os.path.isfile(config.GENOME_IM_PREFIX+'val/'+path):
          impath = config.GENOME_IM_PREFIX+'val/'+path
        elif os.path.isfile(config.GENOME_IM_PREFIX+'test/'+path):
          impath = config.GENOME_IM_PREFIX+'test/'+path
        else:
          impath = config.V7W_IM_PREFIX+path
        return impath

    def _get_single_node(self, att_map, node, iname):
        im = img_as_float(io.imread(iname))[:, :, 0:3]
        if im.ndim == 2: im = im[:, :, np.newaxis].repeat(3, axis=2)
        # im = resize(im, (256,256), mode='reflect')
        h = im.shape[0]
        w = im.shape[1]
        #get image
        # att_map = F.softmax(Variable(att_map.sum(0).data).squeeze().view(1,14*14)).view(14,14).cpu().data.numpy()
        att_map = att_map.squeeze().view(14,14).cpu().data.numpy()
        
        #heatmap = self.colormap.to_rgba(att_map)[:, :, 0:3]
        heatmap = resize(att_map, (h, w), mode='reflect')
        heatmap = self.colormap.to_rgba(heatmap)[:, :, 0:3]
	#get heatmap

        txt = Image.new('RGB', (w,h), (0,0,0))
        d = ImageDraw.Draw(txt)
        words = ''
        for word in node['word']: words += self.vdict_rev[word - 1] + ' '
        d.text((0,0), words, fill=(1,1,1))
        word = np.array(list(txt.getdata())).reshape(h, w, 3)
        #get word
        res = word + heatmap + im
        res_nw = heatmap + im

        return (res / np.max(res)), (res_nw / np.max(res_nw))

    def _get_qustion_str(self, q):
        res = ''
        for i in q:
            if i > 0: res += self.vdict_rev[i - 1] + ' '
        return res.capitalize() + '?'

    def _get_question_img(self, iname, q, ans, predict):
        im = img_as_float(io.imread(iname))[:, :, 0:3]
        if im.ndim == 2: im = im[:, :, np.newaxis].repeat(3, axis=2)
        # im = resize(im, (256,256), mode='reflect')
        h = im.shape[0]
        w = im.shape[1]
        #get image

        txt = Image.new('RGB', (w,h), (0,0,0))
        d = ImageDraw.Draw(txt)
        qstr = self._get_qustion_str(q)+'\n'+self.adict_rev[ans]+'\n'+str(predict)
        wrap_qstr = ''
        for line in textwrap.wrap(qstr, width=80): wrap_qstr += line+'\n'
        print(self._get_qustion_str(q) + ' ' + self.adict_rev[ans])
        d.text((0,0), wrap_qstr, fill=(1,1,1))
        word = np.array(list(txt.getdata())).reshape(h, w, 3)
        #get word
        # res = word + im
        res = im

        return res / np.max(res)

    def _show_single_tree(self, sample_cnt,att_maps, tree, iname, question, ans, predict):
        step = 0
        for i in range(len(tree)):
            if tree[i]['remain'] == 0: continue
            image, imgNW = self._get_single_node(att_maps[i], tree[i], iname)
            # self.writer.add_image('Sample ' + str(sample_cnt), image, step)
            scipy.misc.imsave(self.img_save_prefix+'Sample ' + str(sample_cnt) + '_' + str(step) +'.jpg', image)
            scipy.misc.imsave(self.img_save_prefix+'Sample ' + str(sample_cnt) + '_' + str(step) +'_noWord.jpg', imgNW)
            step += 1
        image = self._get_question_img(iname, question.data, ans.data[0], predict.data)
        # self.writer.add_image('Sample ' + str(sample_cnt), image, step + 1)
        scipy.misc.imsave(self.img_save_prefix+'Sample ' + str(sample_cnt) + '_' + str(step+1) +'.jpg', image)

    def show_node_values(self, node_values, trees, inames, question, ans, predict):
        rmlist = glob(self.img_save_prefix+'*.jpg')
        for file in rmlist: os.remove(file)
        predict = F.softmax(predict).cpu()
        ans = ans.cpu()
        question = question.cpu()
        for i in range(self.sample_num):
            self._show_single_tree(i, node_values[i], trees[i], inames[i], question[i], ans[i], predict[i])
