import cv2
import cupy as cp
from tqdm import tqdm

def calccupy(img, vectorField):
    shape = vectorField.shape
    mapx_base, mapy_base = cp.meshgrid(cp.arange(shape[1]), cp.arange(shape[0]))
    mapx = mapx_base - vectorField[:,:,0]/2
    mapy = mapy_base - vectorField[:,:,1]/2
    return cv2.remap(img, mapx.get().astype(np.float32), mapy.get().astype(np.float32), cv2.INTER_NEAREST)

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
    magnitude = cp.array(magnitude)
    angle = cp.array(angle)
    vectorField = cp.zeros((angle.shape[0], angle.shape[1], 2))
    vectorField[:,:,0] = cp.cos(angle) * magnitude
    vectorField[:,:,1] = cp.sin(angle) * magnitude
    result1 = calccupy(prevFrame,vectorField)
    writer.write(prevFrame)
    writer.write(result1)
    prevFrame = frame.copy()
writer.release()
cap.release()