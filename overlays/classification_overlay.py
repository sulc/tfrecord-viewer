import io
from PIL import Image, ImageDraw, ImageFont


default_color = 'blue'
highlight_color = 'red'


class ClassificationOverlay:
  
  def __init__(self, args):
    self.image_key = args.image_key
    self.class_label_key = args.class_label_key
    self.font = ImageFont.truetype("./fonts/OpenSans-Regular.ttf", 12)


  def apply_overlay(self, image_bytes, example):
    """Apply annotation overlay over input image.
    
    Args:
      image_bytes: JPEG image
      example: TF Example - such as via tf.train.Example().ParseFromString(record)

    Returns:
      image_bytes_with_overlay: JPEG image with annotation overlay.
    """
    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)

    class_label = self.get_label(example.features.feature)

    # draw label over bounding box
    w, h = self.font.getsize(label)
    draw.rectangle((xmin, ymin - 2, xmin + w + 4, ymin - h - 4), fill="white")

    draw.text(
        (xmin + 2, ymin - h - 4),
        label,
        fill=self.bbox_color(label),
        font=self.font,
    )

    with io.BytesIO() as output:
      img.save(output, format="JPEG")
      image_bytes_with_overlay = output.getvalue()

    return image_bytes_with_overlay


  def get_label(self, feature):
    """ From a TF Record Feature, get the image/class label.
    
    Args:
      feature: TF Record Feature
    Returns:
      label (str): image/class
    """
    label = feature[self.class_label_key].bytes_list.value[0].decode("utf-8");
    return label
