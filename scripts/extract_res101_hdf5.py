import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

import os
from PIL import Image
import h5py

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Feature Extractor')
parser.add_argument('--batch_size', type=int, default=256, help="training batch size")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--in_path', type=str, default='data/clevr/CLEVR_v1.0/images/train', help='dir to tensorboard logs')
parser.add_argument('--out_path', type=str, default='data/clevr/clevr_res101/features_train.h5', help='dir to tensorboard logs')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
opt = parser.parse_args()
print(opt)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]) and os.path.getsize(filename)

class DatasetFromFolder(Dataset):

    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()

        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(os.path.join(image_dir, x))]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                  std = [ 0.229, 0.224, 0.225 ]),
        ])

    def __getitem__(self, idx):

        return self.transform(Image.open(self.image_filenames[idx]).convert('RGB')), os.path.basename(self.image_filenames[idx])

    def __len__(self):
        return len(self.image_filenames)

dataloader = torch.utils.data.DataLoader(dataset=DatasetFromFolder(opt.in_path),
                               batch_size=opt.batch_size,
                               num_workers=opt.threads)

net = models.resnet101(pretrained=True).eval()
def printnormpre(self, input):

    global features
    features = input[0].data.cpu().numpy()
net.layer3[-1].relu.register_forward_pre_hook(printnormpre)
net = torch.nn.Sequential(*list(net.children())[:-3])
device = torch.device('cuda' if opt.gpu else "cpu")
net = net.to(device)

if __name__ == '__main__':
    if not os.path.isdir(os.path.dirname(opt.out_path)) : os.mkdir(os.path.dirname(opt.out_path))
    file = h5py.File(opt.out_path,'w')

    print('Extracting features...')
    for input_batch in tqdm(dataloader):

        imgs, names = input_batch
        imgs = imgs.to(device)
        with torch.no_grad() : net(imgs)

        for idx, name in enumerate(names) :
            file.create_dataset(name, data = features[idx])

    file.close()
    print('Extracted feature file : ' + os.path.dirname(opt.out_path))