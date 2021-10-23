from imutils.video import FileVideoStream
from PIL import Image, ImageDraw
from cv2 import flip, resize, imshow, waitKey, rectangle, putText
from time import sleep, time
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

THRESHOLD=0.2

def drawBoundingBoxes(imageData, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    inferenceResults: inference results array
    color: Bounding box color candidates, list of RGB tuples.
    """
    imgHeight, imgWidth, _ = imageData.shape
    thick = int((imgHeight + imgWidth) // 900)
    for res in inferenceResults:
      bbox = res.bbox
      rectangle(imageData,(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color, thick)
      putText(imageData, f'{res.score*100:2.0f}%', (bbox.xmax+10,bbox.ymax), 0, 0.3, (0,255,0))

labels = read_label_file('labels.txt')
interpreter = make_interpreter('efficientdet-lite0-wider_edgetpu.tflite')
interpreter.allocate_tensors()

video_stream = FileVideoStream('./videos/1.mp4').start()
sleep(1.0)

frames = 0
measure_start_ts = time()
while True:
    input_frame = flip(video_stream.read(), 1)
    
    _, scale = common.set_resized_input(
      interpreter, 
      (input_frame.shape[1], input_frame.shape[0]), 
      lambda new_size: resize(input_frame, new_size)
    )
    interpreter.invoke()
    objs = detect.get_objects(interpreter, THRESHOLD, scale)
    
    frames += 1
    nowts = time()
    if nowts - measure_start_ts > 5:
      print(f'FPS: {frames//5}')
      measure_start_ts = nowts
      frames = 0

    drawBoundingBoxes(input_frame, objs, (255, 0, 0))
    imshow("Frame", input_frame)
    key = waitKey(1) & 0xFF
    if key == ord("q"):
      break
