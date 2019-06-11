#!/usr/bin/env python3
import sys
import io
import argparse

import tensorflow as tf
from flask import Flask, render_template, send_file

from overlays import overlay_factory

app = Flask(__name__)

parser = argparse.ArgumentParser(description='TF Record viewer.')
parser.add_argument('tfrecords', type=str, nargs='+',
                    help='path to TF record(s) to view')

parser.add_argument('--image-key', type=str, default="image/encoded",
                    help='Key to the encoded image.')

parser.add_argument('--filename-key', type=str, default="image/filename",
                    help='Key to the unique ID of each record.')

parser.add_argument('--max-images', type=int, default=200,
                    help='Max. number of images to load.')

parser.add_argument('--host', type=str, default="0.0.0.0",
                    help='host/IP to start the Flask server.')

parser.add_argument('--port', type=int, default=5000,
                    help='Port to start the Flask server.')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

parser.add_argument('--overlay', type=str, default="detection",
                    help='Overlay to display. (detection/classification/none)')


#######################################
# Object detection specific arguments #
parser.add_argument('--bbox-name-key', type=str, default="image/object/class/text",
                    help='Key to the bbox label.')

parser.add_argument('--bbox-xmin-key', type=str, default="image/object/bbox/xmin")
parser.add_argument('--bbox-xmax-key', type=str, default="image/object/bbox/xmax")
parser.add_argument('--bbox-ymin-key', type=str, default="image/object/bbox/ymin")
parser.add_argument('--bbox-ymax-key', type=str, default="image/object/bbox/ymax")

parser.add_argument('--coordinates-in-pixels', action="store_true",
                    help='Set if bounding box coordinates are saved in pixels, not in %% of image width/height.')

parser.add_argument('--labels-to-highlight', type=str, default="car",
                    help='Labels for which bounding boxes should be highlighted (red instead of blue).')


###########################################
# Image classification specific arguments #
parser.add_argument('--class-label-key', type=str, default="image/class/text",
                    help='Key to the image class label.')


args = parser.parse_args()


# Variables to be loaded with preload_images()
images = []
filenames = []
captions = []
bboxes = []


def preload_images(max_images):
  """ 
  Load images to be displayed in the browser gallery.

  Args:
    max_images (int): Maximum number of images to load.
  Returns:
    count (int): Number of images loaded.
  """
  count = 0
  overlay = overlay_factory.get_overlay(args.overlay, args)

  for tfrecord_path in args.tfrecords:
    print("Filename: ", tfrecord_path)
    for i, record in enumerate(tf.python_io.tf_record_iterator(tfrecord_path)):
      if args.verbose: print("######################### Record", i, "#########################")
      example = tf.train.Example()
      example.ParseFromString(record)
      feat = example.features.feature

      if len(images) < max_images:
        filename = feat[args.filename_key].bytes_list.value[0].decode("utf-8")
        img =  feat[args.image_key].bytes_list.value[0]
        
        img_with_overlay = overlay.apply_overlay(img, feat)

        filenames.append(filename)
        images.append(img_with_overlay)
        captions.append( tfrecord_path + ":" + filename )
      else:
        return count
      count += 1
  return count



@app.route('/')
def frontpage():
  html = ""
  for i,filename in enumerate(filenames):
    html += '<img data-u="image" src="image/%s" data-caption="%s" />\n' % (i, captions[i])
  return render_template('gallery.html', header=args.tfrecords, images=html)

@app.route('/image/<key>')
def get_image(key):
  """Get image by key (index) from images preloaded when starting the viewer by preload_images().
  """
  key=int(key)
  img = images[key]
  img_buffer = io.BytesIO(img)
  return send_file(img_buffer,
                   attachment_filename=str(key)+'.jpeg',
                   mimetype='image/jpg')

@app.after_request
def add_header(r):
  """
  Add headers to disable Caching,
  (So that images with the same index in different TFRecords are displayed correctly.)
  """
  r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
  r.headers["Pragma"] = "no-cache"
  r.headers["Expires"] = "0"
  r.headers['Cache-Control'] = 'public, max-age=0'
  return r


if __name__ == "__main__":
  print("Pre-loading up to %d examples.." % args.max_images)
  count = preload_images(args.max_images)
  print("Loaded %d examples" % count)
  app.run(host=args.host, port=args.port)

