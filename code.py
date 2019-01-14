import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import operator

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # yellow color mask 
    lower = np.uint8([ 90,   100, 100])
    upper = np.uint8([ 100, 255, 255])
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

def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    #cv2.imshow('white',white_mask)
    # yellow color mask
    lower = np.uint8([0, 190, 190])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    #cv2.imshow('yellow',yellow_mask)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

file_name = 'sample.mov'
cap = cv2.VideoCapture(file_name)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                       qualityLevel = 0.25,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

"""lk_params = dict( winSize  = (5,5),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))"""

# Create some random colors
color = np.random.randint(0,255,(10000,3))
fgbg = cv2.createBackgroundSubtractorKNN()

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
background = old_frame

alpha = 0.1

#buckets to store the angle trajectories
buckets = [0] * 181

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
max1 = 0 
max_angle = 0
x = 0
thresh = 100
cv2.namedWindow('trajectory_information',cv2.WINDOW_NORMAL)
cv2.resizeWindow('trajectory_information', 1000,1000)
#loop through the video until ESC is pressed
while(1):
    x = x + 1
    ret,frame = cap.read()
    fgmask = fgbg.apply(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    #calculate background
    background[fgmask==0] = alpha * frame[fgmask==0] + ( 1- alpha) * background[fgmask==0]
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
   
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #if b < old_frame.shape[0] / 3 :
            #continue
        
        if(abs(b-d)+abs(a-c))<8:
            continue
        
        angle = (math.atan2(c-a,b-d) * 180) / 3.14
        if(angle<0):
            angle = 180 + angle
        #if(int(angle)<=0 or int(angle)>=180):
            #continue
            
        for kk in range(-3,3):
            buckets[int((angle+kk)%180)]+=1
            if buckets[int((angle+kk))%180]>max1:
                max1 = buckets[int((angle+kk)%180)]
                max_angle = (angle + kk)%180
        
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        for j in range(-50,50):  
            mask = cv2.line(mask, (int(a+j),b),(int(c+j),d), [255,255,255], 2)   
    img = cv2.add(mask,frame)
    cv2.imshow('trajectory_information',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    if x % 10 == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    else:
        p0 = good_new.reshape(-1,1,2)
    
    
print('max_angle :-',max_angle)
cv2.destroyAllWindows()

cap.release()


#remove the edges contributed by the foreground objects
frame = background
height  = background.shape[0] 
width = background.shape[1]

alpha1 = 0.01 * max1

only_lanes = select_white_yellow(background)

gray_lanes = cv2.cvtColor(only_lanes, cv2.COLOR_RGB2GRAY)
edges = cv2.GaussianBlur(gray_lanes,(1,1),0)
edges = cv2.Canny(edges,300,1000)
#use the trajectory mask obtained to filter the ROI
gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
#gray_mask = cv2.blur(gray_mask,(5,5))
cv2.imshow('mask',img)
cv2.imwrite('mask.jpg',img)
image1 = cv2.bitwise_and(gray_mask,edges)
filtered = cv2.GaussianBlur(image1,(1,1),0)
cv2.imshow('region_of_interest',filtered)
cv2.imwrite('roi.jpg',filtered)
lines = cv2.HoughLines(filtered,1,np.pi/180,200)

#group similar lines together

y1 = (background.shape[0] * 3)/5
max_distance = 50
lines1=[]
if lines is not None:
        for i in range(0, len(lines)):
             rho = lines[i][0][0]
             theta = lines[i][0][1]
             a = math.cos(theta)
             b = math.sin(theta)
             x = int((rho - y1 * b ) / a)
             lines1.append([rho,theta,x])

def sortSecond(val): 
    return val[2]

print(len(lines1))
lines1.sort(key = sortSecond)
print(lines1)

k = len(lines1)
if lines1 is not None:
            i = 0
            while(i<k):
                j = i+1
                mean = lines1[i]
                count=1
                theta_i = lines1[i][1]
                angle1_i = abs(theta_i) * 180 / 3.14
                angle2_i = int(angle1_i)
                while(j<k):
                    if(abs(lines1[i][2]-lines1[j][2])<max_distance):
                        theta_j = lines1[j][1]
                        angle1_j = abs(theta_j) * 180 / 3.14
                        angle2_j = int(angle1_j)
                        if(abs(angle2_j-angle2_i)<160):
                            mean = list(map(operator.add, mean,lines1[j]))
                        else:
                            j+=1
                            continue
                        count+=1
                        print(angle2_j,'merged with',angle2_i)
                        lines1.pop(j)
                        j-=1
                        k-=1
                    j +=1
                    
                mean = [x / count for x in mean]
                lines1[i] = mean
                i+=1
                    
#cv2.line(frame,(0,int(y1)),(int(background.shape[1]),int(y1)),(255,0,0),3)
                
if lines1 is not None:
        print('Total number of line groups is' ,len(lines1))
        for i in range(0, len(lines1)):
            rho = lines1[i][0]
            theta = lines1[i][1]
            angle1 = abs(theta) * 180 / 3.14
            angle2 = int(angle1)    
            flag = True
            for i in range(-2,2):
               if buckets[(angle2 + i)%180] > alpha1:
                   flag = False
            
            print(angle2)       
            if flag:
                print(angle2,'is dropped')
                continue
            
            if angle2>=75 and angle2<=105:
                print(angle2,'is dropped statically')
                continue
            
            
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            
            x = int((rho - y1 * b ) / a)
            cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            
plt.plot(buckets)
plt.show() 

cv2.imshow('new_image',frame)
cv2.imwrite(str(file_name.split('.')[0])+'_lanes.jpg',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

