import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image


class Datasets(Dataset):
    def __init__(self,path):
        self.path = path
        self.name_x = os.listdir(os.path.join(path,"merge"))
        self.name_y = os.listdir(os.path.join(path,"label_green"))
        self.name_z = os.listdir(os.path.join(path, "label_red"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name_x)

    def __getitem__(self, item):
        namex = self.name_x[item]
        namey = self.name_y[item]
        namez = self.name_z[item]
        img_path = [os.path.join(self.path,i) for i in ("merge","label_green","label_red")]
        img_o = cv2.imread(os.path.join(img_path[0],namex))

        img_1 = cv2.imread(os.path.join(img_path[1],namey))
        img_2 = cv2.imread(os.path.join(img_path[2], namez))
        return self.trans(img_o),self.trans(img_1),namex,self.trans(img_2)



