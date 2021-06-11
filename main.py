import easyocr
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

image=cv2.imread('car.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

bfilter=cv2.bilateralFilter(gray,11,17,17)
edge=cv2.Canny(bfilter,40,200)

keypoints=cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(keypoints)
contours=sorted(contours,key=cv2.contourArea,reverse=True)

location =None
for contour in contours:
    approx=cv2.approxPolyDP(contour,10,True)
    if len(approx)==4:
        location=approx
        break

mask=np.zeros(gray.shape,np.uint8)
new_image=cv2.drawContours(mask,[location],0,255,-1)
new_image=cv2.bitwise_and(image,image,mask=mask)

(x,y)=np.where(mask==255)
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))

cropped_image=gray[x1:x2+1,y1:y2+1]

reader=easyocr.Reader(['en'])
result=reader.readtext(cropped_image)
print(result)

text=result[0][-2]
font=cv2.FONT_HERSHEY_DUPLEX
res=cv2.putText(image,text=text,org=(approx[0][0][0], approx[1][0][1]+60) , fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)


plt.imshow(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
plt.show()








