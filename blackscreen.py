from email.mime import image
import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

#Starting the Webcam
cap= cv2.VideoCapture(1)


#Allowing Webcam to Start by Making Code Sleep for 2 Seconds
time.sleep(2)
bg=[0,0]

#Capturing Background for 60 Frames
for i in range (60):
    ret,bg=cap.read()

#Flipping the Background
bg=np.flip(bg,axis=1)

#Reading Captured Frame Until Camera is Opened
while(cap.isOpened()):
    ret,img=cap.read()
    if not ret:
        break
    img=np.flip(img,axis=1)

#Converting RGB into HSV
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
frame= cv2.resize(frame,(640,480))
image= cv2.resize(image,(640,480))

#Generating Mask to Red Color, These Values can be Changed as per the Color
lower_red=np.array([104,153,70])
upper_red=np.array([104,30,0])

mask_1=cv2.inRange(hsv,lower_red,upper_red)

lower_red=np.array([170,120,70])
upper_red=np.array([180,255,255])

mask_2=cv2.inRange(frame,lower_red,upper_red)

mask_1 = mask_1 + mask_2

#Open and expand image where there is mask 1
mask_1= cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
mask_1= cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

#Selecting Only Part That Does Not Have Mask 1 and Saving in Mask 2
mask_2= cv2.bitwise_not(mask_1)

#Keeping Only Part Of Images Without Red Color
#Or Any Other Color You May Choose
res_1=cv2.bitwise_and(frame,frame,mask=mask_2)
res_2=cv2.bitwise_and(bg,bg,mask=mask_1)

#Generating Final Output by Merging Res 1 and Res 2

final_output= cv2.addWeighted(res_1,1,res_2,1,0)
output_file.write(final_output)
f = frame - res
f = np.where(f == 0 image, f)

#Displaying Output to User
cv2.imshow("magic",final_output)
cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()