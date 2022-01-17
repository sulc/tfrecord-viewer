import io
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pdb

default_color = 'blue'
highlight_color = 'red'


class SemanticSegmentationOverlay:
  
  def __init__(self, args):
    self.segmap_key = args.segmap_key
    self.segmap_format_key = args.segmap_format_key
    self.segmap_colormap_file = args.segmap_colormap_file
    self.font = ImageFont.truetype("./fonts/OpenSans-Regular.ttf", 12)
    self.segmap_raw_divisor_key = args.segmap_raw_divisor_key
    
    if self.segmap_colormap_file is None:
      self.colormap_function = cm.gist_earth
    else:
      self.colormap_function = self.load_colormap()
    


  def apply_overlay(self, image_bytes, example):
    """Apply segmentation overlay over input image.
    
    Args:
      image_bytes: JPEG image.
      feature: TF Record Feature

    Returns:
      image_bytes_with_overlay: JPEG image with segmentation overlay.
    """

    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)

    width, height = img.size

    segmap = self.get_segmap(example, height, width)
    segmap = self.apply_colormap(segmap)
    segmap_img = Image.fromarray(segmap).convert('RGB')

    
    out_img = Image.blend(img, segmap_img, 0.5)


    with io.BytesIO() as output:
      out_img.save(output, format="JPEG")
      image_bytes_with_overlay = output.getvalue()

    return image_bytes_with_overlay


  def load_colormap(self):
      colormap = np.zeros((256, 3), dtype=np.uint8)
      with open(self.segmap_colormap_file, 'rt') as f:
        for i, line in enumerate(f):
          colormap[i] = np.fromstring(line, sep=",", dtype=int)
      listed_colormap = ListedColormap(colormap/255)
      return listed_colormap

  def apply_colormap(self, segmap):
      cm_array = self.colormap_function(segmap/255)
      return np.uint8(cm_array*255)
      
  def get_segmap(self, example, im_height, im_width):
    """ From a TF Record Feature, get the image/class label.
    
    Args:
      feature: TF Record Feature
    Returns:
      mask (numpy.ndarray): image segmentation mask (0-255)
    """
    segmap_format = example.features.feature[self.segmap_format_key].bytes_list.value[0].decode("utf-8")
    example = example.SerializeToString()
    string_feature = tf.io.FixedLenFeature((), tf.string)
    keys_to_features = {self.segmap_key : string_feature, self.segmap_format_key: string_feature}

    parsed_tensors = tf.io.parse_single_example(
        example, features=keys_to_features)

    label_shape = tf.stack([im_height,
          im_width, 1
      ])

    if segmap_format == "raw":
      flattened_label = tf.io.decode_raw(
          parsed_tensors[self.segmap_key], out_type=tf.int32)
      mask = tf.reshape(flattened_label, label_shape).numpy()[:,:,0] // self.segmap_raw_divisor_key
    elif segmap_format == "png":
      label = tf.io.decode_image(parsed_tensors[self.segmap_key], channels=1)
      mask = label.numpy()[:,:,0]
    else:
      raise ValueError("Unknown format: "+segmap_format)
    
    return mask
