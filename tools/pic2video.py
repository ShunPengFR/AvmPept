import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image

def PicToVideo(imgPath, videoPath):
    images = sorted(os.listdir(imgPath))
    fps = 2
    fourcc = VideoWriter_fourcc(*"MJPG")
    im = Image.open(os.path.join(imgPath, images[0]))
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, im.size)
    for im_name in range(len(images)):
        frame = cv2.imread(os.path.join(imgPath, images[im_name]))
        # print(im_name)
        videoWriter.write(frame)
    videoWriter.release()

imgPath = r'F:\data\Boden_AVM_000000_001999\pred40'
videoPath = r'F:\data\Boden_AVM_000000_001999\video\video.avi'
PicToVideo(imgPath, videoPath)