import functools
import operator
import time

import cv2
import numpy as np
import requests

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

threshold = 0.2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth / frameHeight

inHeight = 368
inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

""" uncomment this if you want to see video"""
# vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
#                              (frame.shape[1], frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
k = 0

prev = -1
epsilon = 50
prevXAverage = 0


def isAtLeastOneNotNone(points):
    for point in points:
        if point is not None:
            return True
    return False


lastDetectedTime = -1.0
while 1:

    if time.time() - lastDetectedTime > 3.0:
        prevXAverage = 0

    k += 1
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        prevPoint = None
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # check if hand was detected
    foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)
    xAverage = foldl(operator.add, 0, [x[0] for x in points if x is not None and x[0] is not None])
    numbOfFoundPoints = foldl(operator.add, 0, [1 for x in points if x is not None])
    numbOfFoundPoints = numbOfFoundPoints if numbOfFoundPoints > 0 else 1
    xAverage /= numbOfFoundPoints

    handDetected = isAtLeastOneNotNone(points)
    if handDetected:
        lastDetectedTime = time.time()
        if prevXAverage == 0:
            prevXAverage = xAverage
        print("------------------------------Hand detected\n")

    diff = xAverage - prevXAverage
    print("Diff {}".format(diff))
    if handDetected and (abs(diff) > epsilon):
        if diff > 0:
            print("-------------------------Move LEFT\n")
            requests.get("http://127.0.0.1:5000/previous")
        else:
            print("-------------------------Move RIGHT\n")
            requests.get("http://127.0.0.1:5000/next")
        prevXAverage = xAverage

    """ uncomment this if you want to see drawn skeleton"""
    # for pair in POSE_PAIRS:
    #     partA = pair[0]
    #     partB = pair[1]
    #
    #     if points[partA] and points[partB]:
    #         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
    #         cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    #         cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    print("Time Taken for frame = {}".format(time.time() - t))
    print(".")

    """ uncomment this if you want to see video"""
    # cv2.imshow('Output-Skeleton', frame)
    # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break


    """ uncomment this if you want to see video"""
    # vid_writer.write(frame)

""" uncomment this if you want to see video"""
# vid_writer.release()
