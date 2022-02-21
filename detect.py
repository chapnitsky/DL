# ==============================================================================================
#                                      Imports 
# ==============================================================================================

from cv2 import imread
import numpy as np
import cv2
import torch
from torchvision import transforms as trans
from PIL import Image as PilConvert
from final_project import Net


# ==============================================================================================
#                                   Orgenizing the data
# ==============================================================================================


def Normalize(image, size=128):
    dim = (size, size)
    flipped = cv2.flip(image, 1)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    back = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    F_image = (back.astype(float) * 1.9).astype(np.uint8)
    n_Image = cv2.resize(back, dim, interpolation=cv2.INTER_AREA)
    F_image[F_image >= 255] = 255
    F_image[F_image <= 0] = 0

    return F_image


def getPrediction(N_Image, loaded_model, ImagePath):
    data_transforms = trans.Compose([
        trans.Resize((128, 128)),
        # Random Horizontal Flip
        # transform to tensors
        trans.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # TestImage = Image.open(ImagePath)
    TestImage = PilConvert.fromarray(N_Image)
    loaded_model.eval()
    with torch.no_grad():
        TestImage = data_transforms(TestImage).float()
        # TestImage = TestImage.unsqueeze(0)
        output = loaded_model(TestImage[None, ...])
        # print(output.data)
        _, predicted = torch.max(output.data, 1)
        return predicted


def loadModel(Path, Class_amount=10):
    LoadMod = torch.load(Path)
    model = Net(num_classes=Class_amount).to('cpu')
    model.load_state_dict(LoadMod)
    return model


def SwitchCap(CapVideo, path):
    if CapVideo:
        try:
            # cap= cv2.VideoCapture(0,cv2.CAP_DSHOW)
            cap = cv2.VideoCapture(0)
            print('Swtiching Source to Cam')
        except:
            print('NO CAMERA')
            cap = cv2.VideoCapture(path)

    else:
        print('Swtiching Source to Video')
        cap = cv2.VideoCapture(path)
    return cap


def AppendGesture(img, Gesture, HandPath):
    scale_percent = 150
    x_offset = 30
    y_offset = 40

    try:
        path = HandPath + '' + Gesture + '.png'
        Hand = imread(path, cv2.IMREAD_UNCHANGED)

        width = int(Hand.shape[1] * scale_percent / 100)
        height = int(Hand.shape[0] * scale_percent / 100)
        dim = (width, height)
        # Hand = cv2.resize(Hand, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('',Hand)
        # blended = cv2.addWeighted(img, 0.5, Hand, 0.5, 0)

        x_end = x_offset + Hand.shape[1]
        y_end = y_offset + Hand.shape[0]
        img[y_offset:y_end, x_offset:x_end] = Hand
        return img
    except:
        pass


# =================================================================================
#                                    anlyze video 
# =================================================================================


if __name__ == '__main__':

    # =================================================================================
    #                                    Control Variables 
    # =================================================================================

    imagePath = './Assets/DL_HandGesture.mp4'
    NetPath = './NetPerfomance/98_model_8.pt'
    HAndsDir = './Assets/'

    ClassAmount = 8
    Classes = {'Palm': 0, 'L': 1, 'Fist': 2, 'Fist_Move': 3, 'Thumb': 4, 'Palm_Move': 5, 'C': 6,
               'Down': 7}
    # Classes = {'palm': 0, 'l': 1, 'fist': 2, 'fist_moved': 3, 'thumb': 4, 'index': 5, 'ok': 6, 'palm_moved': 7, 'c': 8,'down': 9}
    pred_Classes = {value: key for (key, value) in Classes.items()}

    # start frame/FPS
    count = 1
    FPs = 4

    # image anlyzation controls
    threshHold = 0.8
    image_size = 128
    feather = 0

    # visualization controls
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    T_Font = 1.25
    org = (30, 30)
    Color = (240, 40, 70)
    thickness = 2
    CapVideo = True
    offset = 20

    # [1] getting the custom  model
    model = loadModel(NetPath, ClassAmount)

    # [2] start the rending of the video
    cap = cv2.VideoCapture(imagePath)

    try:
        while True:

            # [0]
            # update the the capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            count += FPs

            # [1]
            # read and normilze the image
            ret, img = cap.read()
            original = img.copy()
            normalized = Normalize(img)

            # [2]
            # feed and get prediction from the net
            pred = getPrediction(normalized, model, imagePath)

            # [3]
            # plot results

            x = pred.item()
            Gesture = pred_Classes[x]
            Pred_T = 'The class = {}'.format(Gesture)
            Switch_T = 'press space to switch input to webacam'
            Speed_T = 'press A,D in order to speed/slow the video'
            end1 = (int(offset), img.shape[0] - offset)
            end2 = (int(offset), img.shape[0] - 3 * offset)

            # [4]
            # output the predictions
            print('the prediction out of the net : "{}", the class : "{}"'.format(x, pred_Classes[x]))
            cv2.putText(img, Pred_T, org, font, T_Font, (255, 255, 255), thickness + 4, cv2.LINE_AA)
            cv2.putText(img, Pred_T, org, font, T_Font, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(img, Switch_T, end1, font, T_Font - 0.7, (255, 255, 255), thickness + 4, cv2.LINE_AA)
            cv2.putText(img, Switch_T, end1, font, T_Font - 0.7, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(img, Speed_T, end2, font, T_Font - 0.7, (255, 255, 255), thickness + 4, cv2.LINE_AA)
            cv2.putText(img, Speed_T, end2, font, T_Font - 0.7, (0, 0, 0), thickness, cv2.LINE_AA)

            # cv2.putText(img,Pred_T,(x+ (w/2) ,y-font_scale),font,font_scale,NMask_C,thickness,cv2.LINE_AA)
            # cv2.rectangle(img, org, (300,300), Color, 2)

            img = AppendGesture(img, Gesture, HAndsDir)

            cv2.imshow("Output", img)
            # cv2.imshow("Normalized", normalized)
            # cv2.imshow('Original_Input',original)

            # [5]
            # switch case

            k = cv2.waitKey(30) & 0xff
            if k == 97:
                FPs -= 1
                if FPs <= 0:
                    Fps = 1

            elif k == 100:
                FPs += 1
                if FPs > 30:
                    Fps = 30

            elif k == 32:
                cap.release()
                cap = SwitchCap(CapVideo, imagePath)
                CapVideo = not CapVideo

    except:
        print('the video has ended')
