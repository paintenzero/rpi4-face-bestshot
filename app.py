from imutils.video import FileVideoStream
from PIL import Image, ImageDraw
from cv2 import flip, resize, imshow, waitKey, rectangle, putText
from time import sleep, time
from pycoral.utils.dataset import read_label_file
import tflite_runtime.interpreter as tflite 
from pycoral.adapters import common, detect
from sort import Sort
import numpy as np

THRESHOLD=0.3

def drawBoundingBoxes(imageData, trackedResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    trackedResults: a numpy array of detections in the format [[x1,y1,x2,y2,ID],[x1,y1,x2,y2,ID],...]
    color: Bounding box color candidates, list of RGB tuples.
    """
    imgHeight, imgWidth, _ = imageData.shape
    thick = int((imgHeight + imgWidth) // 900)
    for res in trackedResults:
      # print(res[2:4])
      rectangle(imageData, res[0:2], res[2:4], color, thick)
      putText(imageData, f'{res[4]}', (res[0]+10,res[3]-50), 0, 0.3, (0,255,0))

labels = read_label_file('labels.txt')
interpreter = tflite.Interpreter('efficientdet-lite0-wider.tflite')
interpreter.allocate_tensors()

video_stream = FileVideoStream('./videos/1.mp4').start()
sleep(1.0)
tracker = Sort( #create instance of the SORT tracker
  max_age=25, 
  min_hits=10,
  iou_threshold=0.15) 

frames = 0
measure_start_ts = time()
while True:
    input_frame = video_stream.read()
    
    _, scale = common.set_resized_input(
      interpreter, 
      (input_frame.shape[1], input_frame.shape[0]), 
      lambda new_size: resize(input_frame, new_size)
    )
    interpreter.invoke()
    objs = detect.get_objects(interpreter, THRESHOLD, scale)
    
    if len(objs) == 0:
      # If no detections, create empty numpy array 
      dets = np.empty((0, 5))
    else:
      # Create numpy array with detections
      dets = np.empty(shape=(len(objs),5), dtype=np.float32)
      for i in range(len(objs)):
        dets[i] = [objs[i].bbox.xmin, objs[i].bbox.ymin, objs[i].bbox.xmax, objs[i].bbox.ymax, objs[i].score]
    tracked_dets = tracker.update(dets).astype(int)

    # FPS counter
    frames += 1
    nowts = time()
    if nowts - measure_start_ts > 5:
      print(f'FPS: {frames//5}')
      measure_start_ts = nowts
      frames = 0

    if len(tracked_dets) > 0:
      drawBoundingBoxes(input_frame, tracked_dets, (255, 0, 0))
    imshow("Frame", input_frame)
    key = waitKey(1) & 0xFF
    if key == ord("q"):
      break
