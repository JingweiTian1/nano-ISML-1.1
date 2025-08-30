# -*- coding:utf-8 -*-
import os
import torch
from torchvision.utils import save_image
import unet
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch.utils.data import DataLoader
import Bourbon_last_pre
from tqdm import *


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_red = unet.UNet().to(device)
    net_green = unet.UNet().to(device)
    base_pathtt = r"Bourbon_tiqushujut//"
    net_green.load_state_dict(torch.load(r"Bourbon_tiqushuju//fgreen_result2//101.plt"))
    net_red.load_state_dict(torch.load(r"Bourbon_tiqushuju//fred_result2//101.plt"))
    dirlist = os.listdir(r"2-predict-tupian//predict_merge3")
    loader_red = DataLoader(Bourbon_last_pre.Datasets(r"2-predict-tupian","predict_merge3"), batch_size=1, shuffle=True, num_workers=0)####merge原图
    loader_green = DataLoader(Bourbon_last_pre.Datasets(r"2-predict-tupian", "predict_CD313"), batch_size=1,
                            shuffle=True, num_workers=0)
    epoch = 1
    folder_i = 0
    # print(loader.dataset.name_x)
    for name, to_be_predic in tqdm(loader_red, desc=f"Epoch {epoch}/{len(dirlist)}", ascii=True, total=len(loader_red)):
        print(name[0][0:len(name[0])-9])
        folder_i = folder_i + 1
        new_path = r"Bourbon_tiqushuju//output_merge6//" + name[0][0:len(name[0])-9] + "//"
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        input = to_be_predic.to(device)
        with torch.no_grad():
            out_red = net_red(input)
        namex = name[0]
        save_image(out_red, new_path + namex[0:(len(namex) - 5)] + "red.png")
        save_image(input, new_path + namex[0:(len(namex) - 5)] + "yuan.png")
        epoch = epoch + 1

    for name, to_be_predic in tqdm(loader_green, desc=f"Epoch {epoch}/{len(dirlist)}", ascii=True,
                                       total=len(loader_green)):
        print(name[0][0:len(name[0]) - 7])
        folder_i = folder_i + 1
        new_path = r"Bourbon_tiqushuju//output_merge6//" + name[0][0:len(
                name[0]) - 9] + "//"
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        input = to_be_predic.to(device)
        with torch.no_grad():
            out_green = net_green(input)
        namex = name[0]
        save_image(out_green, new_path + namex[0:(len(namex) - 5)] + "green.png")
        save_image(input, new_path + namex[0:(len(namex) - 5)] + "green_yuan.png")
        epoch = epoch + 1













































































        # accuracy = torch.eq(out, fortest).float().mean()
        # print(loss.cpu().numpy())
        # print(name)
    # accuracy = d2l.evaluate_accuracy_gpu(net, loader)
    # print(accuracy)
    # net = torch.load(r"D:\study\TJW\rzunet\dataset\train_img\51.plt")
    # for name in dirlist:
    #     img = cv2.imread(r"D://study//TJW//rzunet//dataset//to_be_predic//"+name)
    #     tran = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #     img = tran(img)
    #     img = img.view(1, 3, 512, 512)
    #     img = img.to(device)
    #     out = net(img)
    #     save_image(out,r"D://study//TJW//rzunet//dataset//predic//"+"pre"+name)
