import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import numpy as np
import random
import string
import matplotlib.pyplot as plt

from utils import generate_random_string, resize_bbox

class CocoDetection(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, resize=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
        self.resize = resize
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        target = target[0]["bbox"]
        shape = img.size[::-1]
        target = resize_bbox(target, shape, self.resize)

        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        
        target = torch.tensor(target)
        return img, target


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




def load_data(batch_size, cocodataset):
      """Load the satellite detection dataset."""
      train_iter = torch.utils.data.DataLoader(cocodataset, batch_size, shuffle=True)
      return train_iter




def get_train_iter(batch_size=32, annotations_file=None, image_folder=None, resize_shape=(300, 300)):
	image_folder     = image_folder
	annotations_file = annotations_file
	cocodataset = CocoDetection(image_folder, annotations_file, resize=resize_shape)
	train_iter = load_data(batch_size, cocodataset)
	return train_iter



def generate_test():
	train_iter = get_train_iter(image_folder="/home/lamaaz/robococo/tmp/airplane/train/", annotations_file="/home/lamaaz/robococo/tmp/airplane/train/_annotations.coco.json")
	batch = next(iter(train_iter))
	# Assuming batch is a tuple containing images and their corresponding labels
	images = batch[0][0:50]  # Selecting the first 20 images from the batch
	labels = batch[1][0:50]  # Selecting the corresponding labels
	reshaped_images = images 
	#reshaped_images = reshaped_images.permute(0, 2, 3, 1)

	for image, label in zip(reshaped_images, labels):
		pil_image = transforms.ToPILImage()(image)
		opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
		x, y, h, w = label.tolist()
		opencv_image = cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		xmin, ymin,  width, height = label
		cv2.imwrite("{}.jpg".format(generate_random_string(10)), opencv_image)


#generate_test()