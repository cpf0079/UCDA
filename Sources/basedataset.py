from torch.utils import data
from PIL import Image
import torch
import numpy as np
from label import get_label
import torchvision.transforms as transforms
from config import cfg


class MyDataset(data.Dataset):
    def __init__(self, txt_dir, label_dir, root, num_samples, num_frames=cfg.NUM_FRAMES, crop_size=cfg.CROP_SIZE):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(txt_dir, 'r')
        imgs = list()
        labels = get_label(label_dir, num_samples)
        for line in fh:
            line = line.rstrip()
            words = line.split()
            vid_name = words[0]
            mos = labels[int(vid_name)]
            mos = np.array(mos)
            mos = torch.Tensor(mos)
            # print(mos)
            imgs.append((vid_name, mos))
        self.num_frames = num_frames
        self.imgs = imgs
        self.root = root
        self.crop_size = crop_size

    def __getitem__(self, index):
        vid_name, label = self.imgs[index]
        label = np.array(label)
        # IMG = np.empty((self.num_frames, self.height, self.width, 3), np.dtype('float32'))
        IMG = np.empty((self.num_frames, self.crop_size[0], self.crop_size[1], 3), np.dtype('float32'))
        for i in range(self.num_frames):
            img = Image.open(self.root + vid_name + '_' + str(i+1) + '.png').convert('RGB')
            img = img.resize(self.crop_size, Image.BICUBIC)
            # img = self.transform(img)
            img = np.asarray(img, np.float32)
            # print(img)
            # img = img[:, :, ::-1]
            # img -= self.mean
            # print(img.shape)
            IMG[i] = img
        IMG = self.normalize(IMG)
        IMG = self.to_tensor(IMG)

        return torch.from_numpy(IMG), torch.from_numpy(label), vid_name

    def __len__(self):
        return len(self.imgs)

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= cfg.IMG_MEAN
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))


if __name__ == '__main__':
    batch_size = 1

    train_data = MyDataset(txt_dir=r'D:\cpf\repository\cpfcpf\UCDA\label.txt',
                           label_dir=r'D:\cpf\repository\VQA\pytorch_version\distribution.xlsx',
                           root='D:\\cpf\\Benchmarks\\VQA\\frames\\')
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    data_iter = iter(train_loader)
    data = data_iter.__next__()
    s_img, s_label, name = data
    print(name)
    print(name[0])
    print(s_img.size())
    print(s_label.size())



