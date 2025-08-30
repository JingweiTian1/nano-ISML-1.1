# -*- coding:utf-8 -*-
import os
import unet
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import data_pre

import torch.nn as nn
import gc

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image



def Train(path,stop_value,save_path,iii):
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = unet.UNet().to(device)
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.BCELoss()
    loader = DataLoader(data_pre.Datasets(path),batch_size=1,shuffle=True,num_workers=0)
    img_save_path = save_path
    epoch = 1
    i = 0
    while True:
        torch.cuda.empty_cache()
        for inputs,green,name in tqdm(loader,desc=f"Epoch {epoch}/{stop_value}",ascii=True,total=len(loader)):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            if iii==1:
                labels = green
            inputs,labels = inputs.to(device),labels.to(device)
            i=i+1
            out = net(inputs)
            loss = loss_func(out,labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            x = inputs[0]
            x_ = out[0]
            y = labels[0]
            img = torch.stack([x,x_,y],0)
            save_image(img.cpu(),os.path.join(img_save_path, name[0]))
        print(f"\nEpoch:{epoch}/{stop_value},Loss:{loss}")
        torch.save(net.state_dict(),os.path.join(img_save_path,f"{epoch}.plt"))
        gc.collect()
        torch.cuda.empty_cache()

        if epoch > stop_value:
            break

        epoch+=1


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    Train("dataset", 100, r'dataset//fgreen_result1',1)
