import os
import unet
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import Bourbon_data_pre3
import torch.nn as nn
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import cv2


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)

        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        return 1 - dice


def dice_loss(y_true, y_pred):
    """
    计算 Dice Loss
    :param y_true: 真实的二值化图像（numpy 数组）
    :param y_pred: 预测的二值化图像（numpy 数组）
    :return: Dice Loss 值
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = np.sum(y_true_flat * y_pred_flat)
    dice_coef = (2. * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))
    dice_loss_value = 1 - dice_coef

    return dice_loss_value

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    print(intersection)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_jaccard_index(img1_path, img2_path):
    img1 = img1_path
    img2 = img2_path
    # 二值化图像
    _, img1_bin = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
    _, img2_bin = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
    # 计算交集和并集
    intersection = np.logical_and(img1_bin, img2_bin)
    union = np.logical_or(img1_bin, img2_bin)
    # 计算交集和并集的像素总数
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    # 计算 Jaccard 相似度
    jaccard_index = intersection_sum / union_sum

    return jaccard_index

def rgb_to_grayscale(image):
    r, g, b = image[0], image[1], image[2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale.unsqueeze(0)

def binarize_image(image, threshold=128):
    # 将阈值转换为 0 到 1 范围内
    threshold = threshold / 255.0
    # 二值化图像
    binary_image = torch.where(image > threshold, torch.tensor(1.0), torch.tensor(0.0))
    return binary_image
def Train(path,stop_value,save_path,iii,yuan,label,redorgreen):
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = unet.UNet().to(device)
    if redorgreen==0:
        net.load_state_dict(torch.load(r"Bourbon_tiqushuju\fred_result2\101.plt"))
    else:
        net.load_state_dict(torch.load(r"Bourbon_tiqushuju\fgreen_result2\101.plt"))
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.BCELoss()
    loader_val = DataLoader(Bourbon_data_pre3.Datasets(path,yuan,label),batch_size=1,shuffle=True,num_workers=0)
    img_save_path = save_path
    epoch = 1
    i = 0
    torch.cuda.empty_cache()
    loss_l = []
    torch.cuda.empty_cache()
    j = 0
    for inputs, green, name in tqdm(loader_val, ascii=True, total=len(loader_val)):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if iii == 1:
            labels = green
        inputs, labels = inputs.to(device), labels.to(device)
        i = i + 1
        out = net(inputs)
        loss = loss_func(out, labels)
        print(loss)
        x = inputs[0]
        x_ = out[0]
        y = labels[0]
        img = torch.stack([x, x_, y], 0)
        os.makedirs(os.path.join(img_save_path,str(j),"p"))
        os.makedirs(os.path.join(img_save_path, str(j), "l"))
        save_image(x_.cpu(), os.path.join(img_save_path,str(j),"p", '1.1.png'))
        save_image(y.cpu(), os.path.join(img_save_path,str(j),"l",'predict_1.1.png'))
        j= j+1
    gc.collect()
    torch.cuda.empty_cache()
    print(loss_l)

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    Train("Bourbon_tiqushuju//test", 100, r'Bourbon_tiqushuju//red_accuracy2',1,"merge","label_red",0)
    Train("Bourbon_tiqushuju//test", 100, r'Bourbon_tiqushuju//green_accuracy2', 1, "green", "label_green", 1)
