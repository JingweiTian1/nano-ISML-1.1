import os
import cv2
import torchvision

from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self,path):
        self.path = path
        self.name_x = os.listdir(os.path.join(path,"last"))
        self.name_y = os.listdir(os.path.join(path,"label"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name_x)

    def __getitem__(self, item):
        namex = self.name_x[item]
        namey = self.name_y[item]
        img_path = [os.path.join(self.path,i) for i in ("last","label")]
        img_o = cv2.imread(os.path.join(img_path[0],namex))
        img_1 = cv2.imread(os.path.join(img_path[1],namey))
        return self.trans(img_o),self.trans(img_1),namex



