import cv2
import os


fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 10
videoWriter = cv2.VideoWriter('3.avi', fourcc, fps, (100, 1000))
L = sorted(os.listdir(os.path.abspath('image/')), key=lambda x:int(x.split('.')[0]))

for i in L:
    fp = os.path.join(os.path.abspath('image/'), i)
    frame = cv2.imread(fp)
    videoWriter.write(frame)
videoWriter.release()

