import zipfile
import os
import PIL
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import detecto
from detecto.core import Model


import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
class objectDetectionTransform:
  def __init__(self):
    self.model = Model()

  def __call__(self, img):
    labels, boxes, scores = self.model.predict(img)  # Get all predictions on an image
    for i,x in enumerate(labels):
      if x == 'bird':
        crop_rectangle = boxes[i].tolist()
        return img.crop(crop_rectangle)
    return img

class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
      iaa.Fliplr(0.5), # horizontal flips
      iaa.Crop(percent=(0, 0.1)), # random crops
      # Small gaussian blur with random sigma between 0 and 0.5.
      # But we only blur about 50% of all images.
      iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.3))
      ),
      # Strengthen or weaken the contrast in each image.
      iaa.LinearContrast((0.75, 1.5)),
      # Add gaussian noise.
      # For 50% of all images, we sample the noise once per pixel.
      # For the other 50% of all images, we sample the noise per pixel AND
      # channel. This can change the color (not only brightness) of the
      # pixels.
      iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
      # Make some images brighter and some darker.
      # In 20% of all cases, we sample the multiplier once per channel,
      # which can end up changing the color of the images.
      iaa.Multiply((0.8, 1.2), per_channel=0.2),
      # Apply affine transformations to each image.
      # Scale/zoom them, translate/move them, rotate them and shear them.
      iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
      )
    ], random_order=True)
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)
    
train_data_transforms = transforms.Compose([
    objectDetectionTransform(),
    transforms.Resize((124, 124)),
    ImgAugTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

val_data_transforms = transforms.Compose([
    objectDetectionTransform(),
    transforms.Resize((124, 124)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


