# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:41:33 2018

@author: N1801255J
"""

import cv2
import numpy as np 
import math
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans 

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    cv2.imshow('white',white_mask)
    # yellow color mask
    lower = np.uint8([100, 190, 190])
    upper = np.uint8([150, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    cv2.imshow('yellow',yellow_mask)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # yellow color mask 
    lower = np.uint8([ 90,   100, 100])
    upper = np.uint8([ 100, 255, 255])
    #lower = np.uint8([ 0,   0, 10])
    #upper = np.uint8([ 255, 255, 100])
    yellow_mask = cv2.inRange(converted, lower, upper)
    cv2.imshow('yellow',yellow_mask)
    # white color mask
    lower = np.uint8([ 0,  190, 0])
    upper = np.uint8([ 255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    cv2.imshow('white',white_mask)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    cv2.imwrite('colour.jpg',mask)
    return cv2.bitwise_and(image, image, mask = mask)


cap = cv2.VideoCapture('UpperBukitTimah_1009_sunny_left.mov')
fgbg = cv2.createBackgroundSubtractorKNN(history=10000)
alpha = 0.1

ret, background = cap.read()

cv2.imwrite('background1.jpg',background)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    background[fgmask==0] = alpha * frame[fgmask==0] + ( 1- alpha) * background[fgmask==0]
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1000,1000)
    cv2.imshow('frame',fgmask)
    cv2.namedWindow('background',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('background', 1000,1000)
    cv2.imshow('background',background)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()

height  = background.shape[0] 
width = background.shape[1]

region_of_interest_vertices = [
    (0, height-200),
    (width / 2, 0),
    (width, height-200),
]

#background = region_of_interest(background, np.array([region_of_interest_vertices], np.int32))
only_lanes = select_white_yellow(background)

#cv2.imshow('background',background)
cv2.imshow('lanes', only_lanes)
cv2.imwrite('background.jpg',background)
cv2.imwrite('background1.jpg',background)

gray_lanes = cv2.cvtColor(only_lanes, cv2.COLOR_RGB2GRAY)
edges = cv2.GaussianBlur(gray_lanes,(21,21),0)
edges = cv2.Canny(gray_lanes,300,1000)

edges[0:int(edges.shape[0]/2)-300,:] = 0 

cv2.imshow('edges',edges)

final_lines = [0] * 18
filtered = cv2.GaussianBlur(edges,(1,1),0)
cv2.imshow('filtered',filtered)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(filtered,(x1,y1),(x2,y2),(0,255,0),2)
    
lines = cv2.HoughLines(filtered,1,np.pi/180,200)


emptylist = []
if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            angle1 = abs(theta) * 180 / 3.14
            print(angle1)
            angle2 = int(angle1)  
            if(angle2>60 and angle2<120):
                continue
            if (final_lines[int(angle2/10)]==0):
                final_lines[int(angle2/10)] = rho
            else :
                final_lines[int(angle2/10)] = (final_lines[int(angle2/10)] + rho)/2
                
            emptylist.append([rho,angle2])
            #print(theta,rho)
            ##if angle2<min1 or angle2>max1:
                #continue
                
            #print(theta,angle2)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            
            cv2.line(background, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
#print (emptylist)
lines1 = np.array(emptylist)
#print(lines1)
#plt.scatter(lines1[:,0],lines1[:,1], label='True Position')   
print(final_lines)
"""
kmeans = KMeans(n_clusters=6)  
kmeans.fit(lines1) 
if kmeans.cluster_centers_ is not None:
    for i in range(0,len(kmeans.cluster_centers_)):
            rho = kmeans.cluster_centers_[i][0]
            theta = kmeans.cluster_centers_[i][1] *3.14 / 180
            angle1 = abs(theta) * 180 / 3.14
            #if(angle1<0):
                #angle1 = 360 + angle1
            angle2 = int(angle1)    
            #print(theta,rho)
            ##if angle2<min1 or angle2>max1:
                #continue
                
            #print(theta,angle2)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            
            cv2.line(background, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            
plt.scatter(lines1[:,0],lines1[:,1], c=kmeans.labels_, cmap='rainbow') 
"""
for i in range(0,18):
    if final_lines[i]!=0:
            theta = i * 10 + 5
            rho = final_lines[i]
            theta = theta * 3.14 / 180 
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            
            #cv2.line(background, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

cv2.imshow('new_image',background)
cv2.imwrite('lanes.jpg',background)
cv2.waitKey(0)
cv2.destroyAllWindows()
    