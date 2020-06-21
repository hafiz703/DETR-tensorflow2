import numpy as np
import os
import cv2 
import random
import warnings
import tensorflow as tf
from pycocotools.coco import COCO
import datetime
from PIL import Image
from utils import *


class Generator(tf.keras.utils.Sequence):
    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=False,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        
         
        config=None
    ):
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()        
        self.config                 = config

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self.groups)

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):         
        return [self.load_annotations(image_index) for image_index in group]

 

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # randomly transform the masks and expand so to have a fake channel dimension
            for i, mask in enumerate(annotations['masks']):
                annotations['masks'][i] = apply_transform(transform, mask, self.transform_parameters)
                annotations['masks'][i] = np.expand_dims(annotations['masks'][i], axis=2)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def box_cxcywh_to_xyxy_scaled(self,x,scale):
      x_c, y_c, w, h = x
      b = [(x_c - 0.5 * w)*scale, (y_c - 0.5 * h)*scale,
          (x_c + 0.5 * w)*scale, (y_c + 0.5 * h)*scale]
       
      return b

   

    def x1y1wh_to_xyxy_scaled(self,box,scale):
      x1,y1,w,h = box
      b = [x1*scale,y1*scale,(x1+w)*scale,(y1+h)*scale]
      return b

    def x1y1wh_to_xyxy(self,box):
      x1,y1,w,h = box
      b = [x1,y1,(x1+w),(y1+h)]
      return b
    
    def box_xyxy_to_cxcywh(self,x):
      x0, y0, x1, y1 = x
      b = [(x0 + x1) / 2, (y0 + y1) / 2,
          (x1 - x0), (y1 - y0)]
      return b 
    
    def cxcywh_to_cxcywh_normalized(self,box,imW,imH):
        assert len(box)==4
        # x,y,w,h = self.box_xyxy_to_cxcywh(box)
        x,y,w,h = box
        ctrX = x/imW
        ctrY = y/imH
        w = w/imW
        h = h/imH
        return  [ctrX,ctrY,w,h]   

    def cocoFormat_to_cxcywh_normalized(self,box,imW,imH):
      xyxy_scaled = self.x1y1wh_to_xyxy(box)
      cxcywh = self.box_xyxy_to_cxcywh(xyxy_scaled)
      cxcywh_normalized = self.cxcywh_to_cxcywh_normalized(cxcywh,imW,imH)
      return cxcywh_normalized


    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)
        
        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)
        new_image_w_h = list(image.shape)[1::-1]
        
        # resize masks
        for i in range(len(annotations['masks'])):
            annotations['masks'][i], _ = self.resize_image(annotations['masks'][i])
            
        # TODO : BBOX image transformation
        
        # apply resizing to annotations too and normalize
        annotations['bboxes'] *= image_scale
        annotations ['scaled_image'] = new_image_w_h #reverse to w,h
      

        # convert to the wanted keras floatx
        image = tf.keras.backend.cast_to_floatx(image)
        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=tf.keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
        
        return image_batch
 

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        for annot in annotations_group:             
            w = annot['scaled_image'][0]
            h = annot['scaled_image'][1]
            annot['bboxes'] = [self.cocoFormat_to_cxcywh_normalized(i,w,h) for i in annot['bboxes']]

                  
 

        return annotations_group

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)       

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets

class CocoGenerator(Generator):
    def __init__(
            self,
            data_dir,
            set_name,
            **kwargs):
        self.data_dir  = data_dir
        self.set_name  = set_name
        self.coco      = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        super(CocoGenerator, self).__init__(**kwargs)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        return len(self.image_ids)   
     

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        return read_image_bgr(path)
    

        

    def load_annotations(self, image_index):
        # get image info
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]

        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = {
            'labels': np.empty((0,)),
            'bboxes': np.empty((0, 4)),
            'masks': [],
            'image_size':[self.coco.loadImgs(self.image_ids[image_index])[0]['width'],self.coco.loadImgs(self.image_ids[image_index])[0]['height']],
            
        }

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
             
            if 'segmentation' not in a:
                raise ValueError('Expected \'segmentation\' key in annotation, got: {}'.format(a))

#             # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate([annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)

            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][2] ,
                a['bbox'][3] ,
            ]]], axis=0)
            
            # // Needs fix
            # mask = np.zeros((image_info['height'], image_info['width'], 1), dtype=np.uint8)
            # for seg in a['segmentation']:
            #     points = np.array(seg).reshape((len(seg) // 2, 2)).astype(int)

            #     # draw mask
            #     cv2.fillPoly(mask, [points.astype(int)], (1,))

            # annotations['masks'].append(mask.astype(float))

        return annotations