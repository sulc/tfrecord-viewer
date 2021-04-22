#!/usr/bin/env python3

"""
Convert tfrecords with image classification datasets to imfolder with images in the format image_folder/classID/filename
"""

import sys
import os
import io
import argparse
from tqdm import tqdm

import tensorflow as tf

parser = argparse.ArgumentParser(description='TF Record viewer.')
parser.add_argument('tfrecords', type=str, nargs='+',
                    help='path to TF record(s) to view')

parser.add_argument('--image-key', type=str, default="image/encoded",
                    help='Key to the encoded image.')

parser.add_argument('--filename-key', type=str, default="image/filename",
                    help='Key to the unique ID of each record.')

parser.add_argument('--class-label-key', type=str, default="image/class/text",
                    help='Key to the image label.')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

parser.add_argument('--output_path', type=str, default="./images_from_tfrecord",
                    help='Path to export images from tfrecords.')


args = parser.parse_args()


def parse_tfrecord(record):
    example = tf.train.Example()
    example.ParseFromString(record)
    feat = example.features.feature

    filename = feat[args.filename_key].bytes_list.value[0].decode("utf-8")
    img =  feat[args.image_key].bytes_list.value[0]
    label = feat[args.class_label_key].bytes_list.value[0].decode("utf-8")

    return filename, img, label

def prepare_class_path(label):
    cls_path = os.path.join(args.output_path,label)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
        if args.verbose:
            print("Creating class directory", cls_path)
    return cls_path

if __name__ == "__main__":
  try:
    tf_record_iterator = tf.python_io.tf_record_iterator
  except:
    tf_record_iterator = tf.compat.v1.python_io.tf_record_iterator
  
  for tfrecord_path in args.tfrecords:
    print("Filename: ", tfrecord_path)
    for i, record in tqdm(enumerate(tf_record_iterator(tfrecord_path)), disable=args.verbose):
        filename, img, label = parse_tfrecord(record)
        cls_path = prepare_class_path(label)
        file_path = os.path.join(cls_path, filename)
        if os.path.exists(file_path):
            print("[Warning] file already exists, overwriting", file_path)
        with open(file_path,'wb') as f:
            f.write(img)