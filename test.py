# -*- coding: utf-8 -*-
import os
import sys

from glob import glob

from PIL import Image

import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

def test(image_directory):
    lst = glob(f"{image_directory}/*.jpg") #glob함수를 이용하여 폴더내의 이미지 파일을 전부 가져옵니다.
    lst.extend(glob(f"{image_directory}/*.png"))
    
    predict_res = [] # 예측값을 저장하는 리스트
    write_string = ""
    
    label = {0: 'paper', 1: 'rock', 2: 'scissors'} # 라벨링을 위한 사전 선언
    test_label = {1:0, 2:1, 0:2} #라벨링의 번호를 지정해주기위한 사전 선언

    
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3) #모델의 가중치를 가져옵니다.
    model.load_state_dict(torch.load('./rps_model.pt', map_location=torch.device("cpu"))) # 그후 가중치 파일을 불러옵니다.
    model.eval() # eval모드 
    
    device = torch.device("cpu") # device선언
    
    for i in lst:

        image = (lambda x : transforms.Compose([transforms.Resize((224, 224)), \
                  transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) \
                  (Image.open(x).convert("RGB")).unsqueeze(0))(i)
        
        
        with torch.no_grad():
          _, preds = torch.max(model(image.to(device)), 1)
        predict_res.append([label[preds.cpu().numpy()[0]] , preds.cpu().numpy()[0]]) #예측 값을 저장합니다.
        
        print(label[preds.cpu().numpy()[0]])

        
    for i in range(len(lst)):
       write_string += f"{os.path.basename(lst[i])} {test_label[predict_res[i][1]]}\n" #txt파일을 저장하기 위해 string값을 계속 쌓아갑니다.
    
    with open("output.txt", "w") as f: #output.txt 파일을 쓰기모드로 저장합니다.
        f.write(write_string) #여태껏 쌓아온 string을 저장합니다.

    
if __name__ == '__main__':
    test("/Users/seok/desktop/test4")
