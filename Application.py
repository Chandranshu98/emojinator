#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from keras.models import load_model
import numpy as np
import os


# In[2]:


model1= load_model('face-rec_256.h5')


# In[3]:


def get_emojis():
    emojis_folder = 'hand_emo/'
    emojis = []
    for emoji in range (len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png',-1))
    return emojis


# In[4]:


def keras_predict(model1,image):
    processed = keras_process_image(image)
    pred_prob=model1.predict(processed)[0]
    pred_class=list(pred_prob).index(max(pred_prob))
    return max(pred_prob),pred_class

def keras_process_image(img):
    x=50
    y=50
    img =cv2.resize(img, (x,y))
    img=np.array(img, dtype=np.float32)
    img=np.reshape(img,(-1,x,y,1))
    return img


# In[5]:


def overlay(image, emoji, x,y,w,h):
    emoji=cv2.resize(emoji, (w,h))
    image[y:y+h, x:x+w]= blend_transparent(image[y:y+h, x:x+w], emoji)
    return image
def blend_transparent(face_img, overlay_t_img):
    overlay_img=overlay_t_img[:,:,:3]
    overlay_mask= overlay_t_img[:,:,3:]
    
    background_mask=255-overlay_mask
    
    overlay_mask= cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask=cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    
    face_part= (face_img *(1.0/255.0)) * (background_mask *(1.0/255.0))
    overlay_part= (overlay_img *(1.0/255.0)) * (overlay_mask *(1.0/255.0))
    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# In[ ]:


emojis=get_emojis()
cap = cv2.VideoCapture(0)
x,y,w,h= 300, 50, 350,350

while (cap.isOpened()):
    ret, img = cap.read()
    img= cv2.flip(img, 1)
    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2= cv2. inRange(hsv, np.array([2,50,60]), np.array([25,150,255]))
    res=cv2.bitwise_and(img,img,mask=mask2)
    gray= cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    median= cv2.GaussianBlur(gray, (5,5),0)
        
    kernel_square=np.ones((5,5), np.uint8)
    dilation =cv2.dilate(median, kernel_square, iterations=2)
    opening= cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        
    ret, thresh= cv2.threshold(opening, 30,255,cv2.THRESH_BINARY)
    thresh= thresh[y:y+h , x:x+w]
    contours= cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(contours)>0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour)>2500:
            x,y,w1,h1 = cv2.boundingRect(contour)
            newImage = thresh[y:y+h1, x:x+w1]
            newImage = cv2.resize(newImage, (50,50))
            pred_prob, pred_class = keras_predict(model1, newImage)
            print(pred_class, pred_prob)
            img= overlay(img, emojis[pred_class], 400, 250, 90, 90)
            
    x,y,w,h= 300, 50, 350, 350
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k=cv2.waitKey(10)
    if k==27:
        break

