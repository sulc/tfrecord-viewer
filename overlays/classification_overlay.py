import io
from PIL import Image, ImageDraw, ImageFont


default_color = 'blue'
highlight_color = 'red'


class ClassificationOverlay:
  
  def __init__(self, args):
    self.class_label_key = args.class_label_key
    self.font = ImageFont.truetype("./fonts/OpenSans-Regular.ttf", 12)


  def apply_overlay(self, image_bytes, feature):
    """Apply annotation overlay over input image.
    
    Args:
      image_bytes: JPEG image.
      feature: TF Record Feature

    Returns:
      image_bytes_with_overlay: JPEG image with annotation overlay.
    """

    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)

    class_label = self.get_label(feature)

    w, h = self.font.getsize(class_label)
    draw.rectangle((10, 10, 14 + w, 10 + h), fill="white")

    draw.text((10, 10), class_label, fill='blue', font=self.font)

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
