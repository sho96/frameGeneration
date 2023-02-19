import cv2
import numpy as np
from tqdm import tqdm

def calcfast(img, vectorField):
    shape = vectorField.shape
    mapx_base, mapy_base = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    mapx = mapx_base - vectorField[:,:,0]/2
    mapy = mapy_base - vectorField[:,:,1]/2
    return cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_NEAREST)

quality = 15

cap = cv2.VideoCapture(r"-----inputVideo-----")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 60 #output fps
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
writer = cv2.VideoWriter(r"result.mp4", fourcc, fps, (1280, 720))

ret, prevFrame = cap.read()

for i in tqdm(range(length-1)):
    ret, frame = cap.read()
    if not ret:
        break
    img1g = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    img2g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1g, img2g, None, 0.5, 3, quality, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    vectorField = np.zeros((angle.shape[0], angle.shape[1], 2))
    vectorField[:,:,0] = np.cos(angle) * magnitude
    vectorField[:,:,1] = np.sin(angle) * magnitude
    result = calcfast(prevFrame,vectorField)
    writer.write(prevFrame)
    writer.write(result)
    prevFrame = frame
writer.release()
cap.release()
