from imutils import resize
from cv2 import cvtColor
from PIL import Image, ImageDraw
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

THRESHOLD=0.4

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

labels = read_label_file('labels.txt')
interpreter = make_interpreter('efficientdet-lite0-wider_edgetpu.tflite')
interpreter.allocate_tensors()

image = Image.open('image.jpg')
_, scale = common.set_resized_input(
  interpreter, 
  image.size, 
  lambda size: image.resize(size, Image.ANTIALIAS)
)

interpreter.invoke()
objs = detect.get_objects(interpreter, THRESHOLD, scale)
draw_objects(ImageDraw.Draw(image), objs, labels)
image.show()
