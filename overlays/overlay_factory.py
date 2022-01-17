
from . import semantic_segmentation_overlay
from . import detection_overlay
from . import classification_overlay


class EmptyOverlay:
  """ Class for empty overlay."""
  def __init__(self, args):
    self.args = args

  def apply_overlay(self, image_bytes, example):
    return image_bytes


overlay_map = {
  'detection': detection_overlay.DetectionOverlay,
  'classification': classification_overlay.ClassificationOverlay,
  'segmentation': semantic_segmentation_overlay.SemanticSegmentationOverlay,
  'none': EmptyOverlay
}

def get_overlay(name, args):
  """ Returns overlay object (by name) initialized by arguments args from tfviewer.py.
  
  Args:
      name (str): Name of the image overlay.
      args
  """
  return overlay_map[name](args)