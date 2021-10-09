import cv2
import matplotlib.pyplot as plt
from PIL import Image
img_path = "IU.jpg"
img = cv2.imread(img_path)

# cv2.rectangle(img,(188,132),(471,522),(0,255,0),3)
cv2.rectangle(img,(471,522),(188,132),(0,255,0),3)



#PIL로 열기

test = 'C:/Users/ddcfd/PycharmProjects/pythonProject/custom_dataset/train/아린/22JVB6VO7X_1.jpg'

im2 = Image.open(test)
im2.show()

# opencv로 열기

cv2.imshow(img_path,img)
cv2.waitKey()
cv2.destroyAllWindows()


# plt로 열기

# plt.figure(figsize=(5,5))
# plt.imshow(img[:,:,::-1])
# plt.show()
