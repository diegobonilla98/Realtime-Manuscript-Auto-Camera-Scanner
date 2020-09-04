import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


images = glob.glob("./test_images/*")
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))

cam = cv2.VideoCapture('./test_images/video.mp4')

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

frame = 0
while True:
    # image = cv2.imread(image_path)
    ret, image = cam.read()
    if not ret:
        break

    if image.shape[0] < image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    original = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (image.shape[1] // 100, image.shape[1] // 100))

    thresh = 255 - cv2.inRange(hsv, (80, 0, 150), (160, 70, 230))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 50, image.shape[1] // 50))
                              , iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 10, image.shape[1] // 10))
                              , iterations=2)
    thresh = 255 - thresh
    thresh = cv2.Canny(thresh, 10, 50, apertureSize=7)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 50, image.shape[1] // 50))
                              , iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not len(contours):
        continue
    main_contour = max(contours, key=len)

    bbox = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(bbox)
    for vertex in box:
        cv2.circle(image, tuple(np.int32(vertex)), image.shape[1] // 100, (255, 0, 0), -1)

    warped = four_point_transform(original, box)

    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    warped_hsv[:, :, 2] = clahe.apply(warped_hsv[:, :, 2])
    warped = cv2.cvtColor(warped_hsv, cv2.COLOR_HSV2BGR)

    warped = np.uint8(np.clip(1.7 * np.float32(warped) - 100, 0, 255))
    warped = cv2.bilateralFilter(warped, 9, 75, 75)
    warped = cv2.detailEnhance(warped, sigma_s=3, sigma_r=0.15)

    canvas = np.zeros((max([original.shape[0], warped.shape[0] + thresh.shape[0]]),
                       original.shape[1] + max([warped.shape[1] + thresh.shape[1]]), 3), np.uint8)

    canvas[:original.shape[0], :original.shape[1]] = original
    canvas[:warped.shape[0], original.shape[1]: max([warped.shape[1] + thresh.shape[1]])] = warped
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    canvas[warped.shape[0]: canvas.shape[0], original.shape[1]: original.shape[1] + thresh.shape[1]] = thresh

    canvas = canvas[:950, :730]

    if frame == 0:
        h, w = canvas.shape[:2]
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))

    out.write(canvas)

    frame = frame + 1
    cv2.imshow("Result", canvas)
    if cv2.waitKey(1) == ord('q') or frame > 400:
        out.release()
        cam.release()
        cv2.destroyAllWindows()
        break

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('CamScannerBarato by Diego Bonilla')
    #
    # ax1.imshow(original[:, :, ::-1])
    # ax1.axis('off')
    # ax2.imshow(warped[:, :, ::-1])
    # ax2.axis('off')
    #
    # plt.show()
