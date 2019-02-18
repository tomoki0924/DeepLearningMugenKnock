import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 572, 572#572, 572
out_height, out_width = 388, 388#388, 388
GPU = False
torch.manual_seed(0)


def crop_layer(layer, size):
    _, _, h, w = layer.size()
    _, _, _h, _w = size
    print(h, _h)
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)
    return layer[:, :, ph:ph+_h, pw:pw+_w]

    
class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        self.enc1 = []
        for i in range(2):
            f = 3 if i == 0 else 64
            self.enc1.append(torch.nn.Conv2d(f, 64, kernel_size=3, padding=0, stride=1))
        
        self.enc2 = []
        for i in range(2):
            f = 64 if i == 0 else 128
            self.enc2.append(torch.nn.Conv2d(f, 128, kernel_size=3, padding=0, stride=1))

        self.enc3 = []
        for i in range(2):
            f = 128 if i == 0 else 256
            self.enc3.append(torch.nn.Conv2d(f, 256, kernel_size=3, padding=0, stride=1))
        
        self.enc4 = []
        for i in range(2):
            f = 256 if i == 0 else 512
            self.enc4.append(torch.nn.Conv2d(f, 512, kernel_size=3, padding=0, stride=1))
            
        self.enc5 = []
        for i in range(2):
            f = 512 if i == 0 else 1024
            self.enc5.append(torch.nn.Conv2d(f, 1024, kernel_size=3, padding=0, stride=1))

        self.tconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = []
        for i in range(2):
            f = 1024 if i == 0 else 512
            self.dec4.append(torch.nn.Conv2d(f, 512, kernel_size=3, padding=0, stride=1))

        self.tconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = []
        for i in range(2):
            f = 512 if i == 0 else 256
            self.dec3.append(torch.nn.Conv2d(f, 256, kernel_size=3, padding=0, stride=1))

        self.tconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = []
        for i in range(2):
            f = 256 if i == 0 else 128
            self.dec2.append(torch.nn.Conv2d(f, 128, kernel_size=3, padding=0, stride=1))

        self.tconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = []
        for i in range(2):
            f = 128 if i == 0 else 64
            self.dec1.append(torch.nn.Conv2d(f, 64, kernel_size=3, padding=0, stride=1))

        self.out = torch.nn.Conv2d(64, num_classes+1, kernel_size=1, padding=0, stride=1)
        
        
    def forward(self, x):
        # block conv1
        for layer in self.enc1:
            x = F.relu(layer(x))
        x_enc1 = x
        x = F.max_pool2d(x_enc1, 2, stride=2, padding=0)
        
        # block conv2
        for layer in self.enc2:
            x = F.relu(layer(x))
        x_enc2 = x
        x = F.max_pool2d(x_enc2, 2, stride=2, padding=0)
        
        # block conv3
        for layer in self.enc3:
            x = F.relu(layer(x))
        x_enc3 = x
        x = F.max_pool2d(x_enc3, 2, stride=2, padding=0)
        
        # block conv4
        for layer in self.enc4:
            x = F.relu(layer(x))
        x_enc4 = x
        x = F.max_pool2d(x_enc4, 2, stride=2, padding=0)
        
        # block conv5
        for layer in self.enc5:
            x = F.relu(layer(x))
        print(x.size())
        x = self.tconv4(x)
        print(x.size())
        _x = crop_layer(x_enc4, x.size())
        print(_x.size())
        x = torch.cat((_x, x), dim=1)
        for layer in self.dec4:
            x = F.relu(layer(x))

        x = self.tconv3(x)
        _x = crop_layer(x_enc3, x.size())
        x = torch.cat((_x, x), dim=1)
        for layer in self.dec3:
            x = F.relu(layer(x))

        x = self.tconv2(x)
        _x = crop_layer(x_enc2, x.size())
        x = torch.cat((_x, x), dim=1)
        for layer in self.dec2:
            x = F.relu(layer(x))

        x = self.tconv1(x)
        _x = crop_layer(x_enc1, x.size())
        x = torch.cat((_x, x), dim=1)
        for layer in self.dec1:
            x = F.relu(layer(x))

        x = self.out(x)
        
        return x

CLS = {'akahara': [0,0,128],
       'madara': [0,128,0]}
    
# get train data
def data_load(path, hf=False, vf=False):
    xs = np.ndarray((0, img_height, img_width, 3))
    ts = np.ndarray((0, out_height, out_width))
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            gt = cv2.imread(gt_path)
            gt = cv2.resize(gt, (out_width, out_height), interpolation=cv2.INTER_NEAREST)

            t = np.zeros((out_height, out_width), dtype=np.int)

            for i, (_, vs) in enumerate(CLS.items()):
                ind = (gt[...,0] == vs[0]) * (gt[...,1] == vs[1]) * (gt[...,2] == vs[2])
                t[ind] = i + 1
            #print(gt_path)
            #import matplotlib.pyplot as plt
            #plt.imshow(t)
            #plt.show()

            ts = np.r_[ts, t[None, ...]]
            
            paths.append(path)

            if hf:
                xs = np.r_[xs, x[:, ::-1][None, ...]]
                ts = np.r_[ts, t[:, ::-1][None, ...]]
                paths.append(path)

            if vf:
                xs = np.r_[xs, x[::-1][None, ...]]
                ts = np.r_[ts, t[::-1][None, ...]]
                paths.append(path)

            if hf and vf:
                xs = np.r_[xs, x[::-1, ::-1][None, ...]]
                ts = np.r_[ts, t[::-1, ::-1][None, ...]]
                paths.append(path)

    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths


# train
def train():
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = Mynet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 16
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(500):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)

        print(y.size())

        y = y.view(-1, num_classes+1)
        t = t.view(-1)

        print(y.size(), t.size())
        
        y = F.log_softmax(y, dim=1)
        loss = torch.nn.CrossEntropyLoss()(y, t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = Mynet().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    xs, ts, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
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
