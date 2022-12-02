import cv2
# from tracker import *

# Create tracker for obj

# tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('video_2022-11-25_03-06-10.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100)

while (True):

    # specifing area
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # print(height, width)

    roi = frame[90:700, 0:1250]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 254, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    # object Tracking
    # boxes_ids = tracker.update(detections)
    # for bi in boxes_ids:
    #     x, y, w, h, id = bi
    #     cv2.putText(roi, str(id), (x, y-15),
    #                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    #     cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # cv2.imshow('roi', roi)
    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
