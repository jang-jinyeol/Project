import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


class FaceDetectorClass(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn, classifier):
        self.mtcnn = mtcnn
        self.classifier = classifier

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        for box, prob, ld in zip(boxes, probs, landmarks):
            # Draw rectangle on frame
            cv2.rectangle(frame,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255),
                          thickness=2)

            # Show probability
            # cv2.putText(frame, str(
            #        prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)

        return frame

    def _detect_ROIs(self, boxes):
        """
        Return ROIs as a list
        (X1,X2,Y1,Y2)
        """
        ROIs = list()
        for box in boxes:
            ROI = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            ROIs.append(ROI)

        return ROIs

    def _blur_face(self, image, factor=3.0):
        """
        Return the blured image
        """
        # Determine size of blurring kernel based on input image
        (h, w) = image.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)

        # Ensure width and height of kernel are odd
        if kW % 2 == 0:
            kW -= 1

        if kH % 2 == 0:
            kH -= 1

        # Apply a Gaussian blur to the input image using our computed kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)

    def _is_it_Rosa(self, face):
        destRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(destRGB.astype('uint8'), 'RGB')

        preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

        processed_img = preprocess(PIL_img)
        batch_t = torch.unsqueeze(processed_img, 0)
        with torch.no_grad():
            out = self.classifier(batch_t)
            _, pred = torch.max(out, 1)

        prediction = np.array(pred[0])

        # Rosa = 0, not_Rosa=1
        if prediction == 0:
            return ('★jinyeol★')
        else:
            return ('nope')

    def run(self, blur_setting=True):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture('http://192.168.0.75:4747/mjpegfeed?640x480')

        while True:
            ret, frame = cap.read()
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                self._draw(frame, boxes, probs, landmarks)

                # Blur face if Rosa
                if blur_setting == True:

                    # Extract the face ROI
                    ROIs = self._detect_ROIs(boxes)

                    for roi in ROIs:
                        (startY, endY, startX, endX) = roi
                        face = frame[startY:endY, startX:endX]
                        # run the classifier on bounding box
                        pred = self._is_it_Rosa(face)
                        print(pred)

                        if pred == '★jinyeol★':
                            blured_face = self._blur_face(face)
                            frame[startY:endY, startX:endX] = blured_face

                        else:
                            pass


            except:
                pass

            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
