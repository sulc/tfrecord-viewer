import io
from PIL import Image, ImageDraw, ImageFont


default_color = 'blue'
highlight_color = 'red'


class DetectionOverlay:

  def __init__(self, args):
    self.args = args
    self.labels_to_highlight = args.labels_to_highlight.split(";")
    try:
      self.font = ImageFont.truetype("./fonts/OpenSans-Regular.ttf", 12)
    except OSError:
      self.font = ImageFont.load_default()


  def apply_overlay(self, image_bytes, example):
    """Apply annotation overlay over input image.

    Args:
      image_bytes: JPEG image
      example: TF Example - such as via tf.train.Example().ParseFromString(record)

    Returns:
      image_bytes_with_overlay: JPEG image with annotation overlay.
    """
  
    bboxes = self.get_bbox_tuples(example.features.feature)
    image_bytes_with_overlay = self.draw_bboxes(image_bytes, bboxes)
    return image_bytes_with_overlay


  def get_bbox_tuples(self, feature):
    """ From a TF Record Feature, get a list of tuples representing bounding boxes

    Args:
      feature: TF Record Feature
    Returns:
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]
    """
    bboxes = []
    if self.args.bbox_name_key in feature:
      for ibbox, label in enumerate (feature[self.args.bbox_name_key].bytes_list.value):
        bboxes.append( (label.decode("utf-8"),
                        feature[self.args.bbox_xmin_key].float_list.value[ibbox],
                        feature[self.args.bbox_xmax_key].float_list.value[ibbox],
                        feature[self.args.bbox_ymin_key].float_list.value[ibbox],
                        feature[self.args.bbox_ymax_key].float_list.value[ibbox]
      ) )
    else:
      print("Bounding box key '%s' not present." % (self.args.bbox_name_key))
    return bboxes

  def bbox_color(self, label):
    if label in self.labels_to_highlight:
      return highlight_color
    else:
      return default_color

  def bboxes_to_pixels(self, bbox, im_width, im_height):
    """
    Convert bounding box coordinates to pixels.
    (It is common that bboxes are parametrized as percentage of image size
    instead of pixels.)

    Args:
      bboxes (tuple): (label, xmin, xmax, ymin, ymax)
      im_width (int): image width in pixels
      im_height (int): image height in pixels

    Returns:
      bboxes (tuple): (label, xmin, xmax, ymin, ymax)
    """
    if self.args.coordinates_in_pixels:
      return bbox
    else:
      label, xmin, xmax, ymin, ymax = bbox
      return [label, xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]

  def draw_bboxes(self, image_bytes, bboxes):
    """Draw bounding boxes onto image.

    Args:
      image_bytes: JPEG image.
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]

    Returns:
      image_bytes: JPEG image including bounding boxes.
    """
    img = Image.open(io.BytesIO(image_bytes))

    draw = ImageDraw.Draw(img)

    width, height = img.size

    for bbox in bboxes:
      label, xmin, xmax, ymin, ymax = self.bboxes_to_pixels(bbox, width, height)
      draw.rectangle([xmin, ymin, xmax, ymax], outline=self.bbox_color(label))

      w, h = self.font.getsize(label)
      draw.rectangle((xmin, ymin, xmin + w + 4, ymin + h), fill="white")

      draw.text((xmin+4, ymin), label, fill=self.bbox_color(label), font=self.font)

    with io.BytesIO() as output:
      if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
      img.save(output, format="JPEG")
      output_image = output.getvalue()
    return output_image
