import pixellib
from pixellib.instance import instance_segmentation
import cv2

#checkpoint github url 
#https://github.com/matterport/Mask_RCNN/releases   2   H.5



#setting up model
segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5')

url = "http://192.168.1.245:8080/video"
cap = cv2.VideoCapture(url)

while cap.isOpened():
    ret, frame = cap.read()

    res = segmentation_model.segmentFrame(frame, show_bboxes=True)
    image = res[1]
    temp = cv2.flip(image, 1)

    cv2.imshow("Instance segmentation", temp)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
