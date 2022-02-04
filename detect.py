




#==============================================================================================
#                                      Imports 
#==============================================================================================

from importlib.resources import path
from msilib.schema import Class
from turtle import back

import pandas as pd
import numpy as np
import cv2
from scipy.spatial import distance


import torch 
import torchvision
from torchvision import transforms as trans

import matplotlib.pyplot as plt
import seaborn as sns

import PIL.Image as Image
from PIL import Image as PilConvert
from final_class import Net


#==============================================================================================
#                                   Orgenizing the data
#==============================================================================================


def Normalize(image,size= 128):
    dim = (size,size)
    # flipped = cv2.flip(image,1)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # back = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_Image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    
    return n_Image


def getPrediction(N_Image,loaded_model,ImagePath):
    
    # image_Tensore = torch.from_numpy(N_Image)
    # image_Tensore = loader(N_Image).float()
    
    
    # test_loader = torch.utils.data.DataLoader(
    #     [N_Image],
    #     batch_size=1,
    #     num_workers=0,
    #     shuffle=False
    # )
    data_transforms = trans.Compose([
        trans.Resize((128,128)),
        # Random Horizontal Flip
        # transform to tensors
        trans.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # TestImage = Image.open(ImagePath)
    TestImage =PilConvert.fromarray(N_Image)
    loaded_model.eval()
    with torch.no_grad():
        TestImage = data_transforms(TestImage).float()
        # TestImage = TestImage.unsqueeze(0)
        output  = loaded_model(TestImage[None,...])
        _, predicted = torch.max(output.data, 1)
        return predicted
        # for data in test_loader:
        #     data = data.to('cpu')
        #     # Get the predicted classes for this batch
        #     output = model(data)
        #     # Calculate the loss for this batch
            
        #     _, predicted = torch.max(output.data, 1)
        #     return predicted
    
    
    # return average loss for the epoch
   


def loadModel(Path,Class_amount = 10):
    LoadMod  = torch.load(Path)
    model = Net(num_classes=Class_amount).to('cpu')
    model.load_state_dict(LoadMod)
    return model


def SwitchCap(CapVideo,path):
    if CapVideo:
        try:
            #cap= cv2.VideoCapture(0,cv2.CAP_DSHOW)
            cap= cv2.VideoCapture(0)
            print('Swtiching Source to Cam')
        except:
            print('NO CAMERA')
            cap =cv2.VideoCapture(path)

    else:
        print('Swtiching Source to Video')
        cap =cv2.VideoCapture(path)
    return cap

#=================================================================================
#                                    anlyze video 
#=================================================================================


if __name__ =='__main__':

    
    #=================================================================================
    #                                    Control Variables 
    #=================================================================================



    
    imagePath = './Assets/video_test.mp4'
    NetPath = './NetPerfomance/98_model_8.pt'    


    ClassAmount = 8
    Classes= {'palm': 0, 'l': 1, 'fist': 2, 'fist_moved': 3, 'thumb': 4, 'palm_moved': 5, 'c': 6,
           'down': 7}
    # Classes = {'palm': 0, 'l': 1, 'fist': 2, 'fist_moved': 3, 'thumb': 4, 'index': 5, 'ok': 6, 'palm_moved': 7, 'c': 8,'down': 9}
    pred_Classes = {value : key for (key, value) in Classes.items()}



    

    # start frame/FPS
    count = 1
    FPs = 4


    #image anlyzation controls
    threshHold = 0.8
    image_size = 128
    feather = 0

    #visualization controls
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    T_Font = 1.25
    org = (30, 30)
    Color = (240,40,70) 
    thickness = 3
    CapVideo = True



    #[1] getting the custom  model 
    model = loadModel(NetPath,ClassAmount)


    #[2] start the rending of the video
    cap = cv2.VideoCapture(imagePath)

    while True:
        
        #[0]
        #update the the capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        count+=FPs

        #[1]
        #read and normilze the image
        ret, img = cap.read()
        original = img.copy()
        normalized = Normalize(img)
        


        #[2]
        #feed and get prediction from the net
        pred = getPrediction(normalized,model,imagePath)
        
        
        #[3]
        #plot results 

        x = pred.item()
        Pred_T = 'The class = {}'.format(pred_Classes[x])
        
        
        #[4]
        #output the predictions
        print('the prediction out of the net : "{}", the class : "{}"'.format(x, pred_Classes[x]))
        cv2.putText(img, Pred_T, org, font, T_Font, Color, thickness, cv2.LINE_AA)
                #cv2.putText(img,Pred_T,(x+ (w/2) ,y-font_scale),font,font_scale,NMask_C,thickness,cv2.LINE_AA)
        # cv2.rectangle(img, org, (300,300), Color, 2)
                

        cv2.imshow('Original_Input',original)
        cv2.imshow("Output",img)
        cv2.imshow("Normalized", normalized)
        
        
        #[5]
        #switch case
        
        k= cv2.waitKey(30) & 0xff
        if k== 97: 
            FPs-=1
            if FPs<=0 :
                Fps = 1

        elif k==100:
            FPs+=1
            if FPs>30:
                Fps = 30 
                
        elif k == 32:
            cap.release()
            cap =SwitchCap(CapVideo,imagePath) 
            CapVideo = not CapVideo
            
    
    cap.release()
    cv2.destroyAllWindows()
        
        
        
    