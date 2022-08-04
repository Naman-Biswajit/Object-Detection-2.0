import cv2 as cv


async def YoloDetect(capture, labels, network):
    status, frame = capture.read()
    blob = cv.dnn.blobFromImage(
        frame, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    network.setInput(blob)

    cv.imshow('frame', frame)
    cv.waitKey(1)
