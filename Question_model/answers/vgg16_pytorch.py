import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy
from collections import OrderedDict
from tqdm import tqdm

# class config
class_label = ['akahara', 'madara']
class_N = len(class_label)

# config
img_height, img_width = 128, 128
channel = 3

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# other
model_path = 'vgg16.pt'

torch.manual_seed(0)

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(channel, 64, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        
        self.fc1 = torch.nn.Linear(int((img_height / 32 * img_width / 32 * 512)), 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc_out = torch.nn.Linear(4096, class_N)
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
    
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.nn.Dropout()(x)
        x = F.relu(self.fc2(x))
        x = torch.nn.Dropout()(x)
        x = self.fc_out(x)
        x = F.softmax(x, dim=1)
        
        return x



# get train data
def data_load(path, hf=False, vf=False, rot=False):
    if rot == 0:
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    ts = []
    
    data_num = 0
    for dir_path in glob(path + '/*'):
        data_num += len(glob(dir_path + "/*"))
            
    pbar = tqdm(total = data_num)
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            for i, cls in enumerate(class_label):
                if cls in path:
                    t = i

            paths.append({'path': path, 'hf': False, 'vf': False, 'rot': 0})
            ts.append(t)

            # horizontal flip
            if hf:
                paths.append({'path': path, 'hf': True, 'vf': False, 'rot': 0})
                ts.append(t)
            # vertical flip
            if vf:
                paths.append({'path': path, 'hf': False, 'vf': True, 'rot': 0})
                ts.append(t)
            # horizontal and vertical flip
            if hf and vf:
                paths.append({'path': path, 'hf': True, 'vf': True, 'rot': 0})
                ts.append(t)
            # rotation
            if rot is not False:
                angle = rot
                while angle < 360:
                    paths.append({'path': path, 'hf': False, 'vf': False, 'rot': rot})
                    angle += rot
                    ts.append(t)
                
            pbar.update(1)
                    
    pbar.close()
    
    return np.array(paths), np.array(ts)

def get_image(infos):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        
        # channel BGR -> Gray
        if channel == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        # channel BGR -> RGB
        if channel == 3:
            x = x[..., ::-1]

        # normalization [0, 255] -> [-1, 1]
        x = x / 127.5 - 1

        # horizontal flip
        if hf:
            x = x[:, ::-1]

        # vertical flip
        if vf:
            x = x[::-1]

        # rotation
        scale = 1
        _h, _w, _c = x.shape
        max_side = max(_h, _w)
        tmp = np.zeros((max_side, max_side, _c))
        tx = int((max_side - _w) / 2)
        ty = int((max_side - _h) / 2)
        tmp[ty: ty+_h, tx: tx+_w] = x.copy()
        M = cv2.getRotationMatrix2D((max_side / 2, max_side / 2), rot, scale)
        _x = cv2.warpAffine(tmp, M, (max_side, max_side))
        _x = _x[tx:tx+_w, ty:ty+_h]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)
    
    if channel == 1:
        xs = np.expand_dims(xs, axis=-1)
    
    xs = np.transpose(xs, (0,3,1,2))
    
    return xs


# train
def train():
    # model
    model = VGG16().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    paths, ts = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 32
    mbi = 0
    data_N = len(paths)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_func = torch.nn.NLLLoss()
    
    for i in range(500):
        if mbi + mb > data_N:
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (data_N - mbi))]))
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        x = torch.tensor(get_image(paths[mb_ind]), dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)

        loss = loss_func(torch.log(y), t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
        if (i + 1) % 50 == 0:
            print("iter >>", i+1, ', loss >>', loss.item(), ', accuracy >>', acc)

    torch.save(model.state_dict(), model_path)

# test
def test():
    # model
    model = VGG16().to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    paths, ts = data_load('../Dataset/test/images/', hf=False, vf=False, rot=False)

    with torch.no_grad():
        for i in range(len(paths)):
            path = paths[i]
            x = get_image(paths)
            t = ts[i]
            
            x = torch.tensor(x, dtype=torch.float).to(device)
            
            pred = model(x)
            pred = F.softmax(pred, dim=1).detach().cpu().numpy()[0]
        
            print("in {}, predicted probabilities >> {}".format(path, pred))
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
