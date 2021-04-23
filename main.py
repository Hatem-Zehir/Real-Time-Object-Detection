import cv2

# Threshold to detect object
Threshold = 0.6

cap = cv2.VideoCapture(0)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = Threshold)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(), confs.flatten(), bbox):
            
            conf = [str(round(confidence*100, 2)), "%"]
            conf = ''.join(conf)
            
            cv2.rectangle(img, box, color=(255, 0, 0), thickness=3)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, conf, (box[2]-80, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("CAMERA", img)

    # Stop if ESC key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()