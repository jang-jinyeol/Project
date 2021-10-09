import cv2
from facenet_pytorch import  MTCNN

mtcnn=MTCNN(device='cuda')


class FaceDetector(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn
    def run(self):
        img_path = "IU.jpg"
        img = cv2.imread(img_path)

        boxes, probs, landmark =self.mtcnn.detect(img, landmarks=True)
        cv2.rectangle(img,(int(boxes[0][2]),int(boxes[0][3])),(int(boxes[0][0]),int(boxes[0][1])),(0,255,0),3)
        
'''
# bb와 putText 그리는 함수
def makeBox(voc_im,bbox,objects):
  image = voc_im.copy()
  for i in range(len(objects)):
  cv2.rectangle(image,(int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),color = (0,255,0),thickness = 1)
  cv2.putText(image, objects[i], (int(bbox[i][0]), int(bbox[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) # 크기, 색, 굵기
  return image


'''

        # cv2.rectangle(img, (471, 522), (188, 132), (0, 255, 0), 3)
        # print(int(boxes[0][2]))
        # print(boxes[0][3])
        # print(boxes[0][1])
        # print(boxes[0][2])


        # print(boxes)
        # print(boxes[0][3])


        cv2.imshow(img_path, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


fcd = FaceDetector(mtcnn)
fcd.run()
