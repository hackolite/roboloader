from typing import Tuple, Optional, Callable
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import numpy as np
import random
import string
import matplotlib.pyplot as plt

from .utils import generate_random_string, resize_bbox


class CocoDetection(torch.utils.data.Dataset):
    """Classe représentant le jeu de données MS Coco Detection.

    Args:
        root (str): Répertoire racine où les images sont téléchargées.
        annFile (str): Chemin vers le fichier d'annotations JSON.
        transform (callable, optionnel): Une fonction/transformation qui prend une image PIL
            et retourne une version transformée. Par exemple, ``transforms.ToTensor``.
        target_transform (callable, optionnel): Une fonction/transformation qui prend la
            cible et la transforme.
    """
    
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        resize: Optional[Tuple[int, int]] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.resize = resize
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Indice

        Returns:
            tuple: Tuple (image, target). La cible est l'objet retourné par ``coco.loadAnns``.
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

        target = torch.tensor(target)
        return img, target

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        fmt_str = 'Jeu de données ' + self.__class__.__name__ + '\n'
        fmt_str += '    Nombre de points de données : {}\n'.format(self.__len__())
        fmt_str += '    Répertoire racine : {}\n'.format(self.root)
        tmp = '    Transformations (si disponibles) : '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Transformations cibles (si disponibles) : '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def load_data(batch_size: int, cocodataset: CocoDetection) -> torch.utils.data.DataLoader:
    """Charger le jeu de données de détection satellite."""
    train_iter = torch.utils.data.DataLoader(cocodataset, batch_size, shuffle=True)
    return train_iter

def get_train_iter(
    batch_size: int = 32,
    annotations_file: Optional[str] = None,
    image_folder: Optional[str] = None,
    resize_shape: Tuple[int, int] = (300, 300)
) -> torch.utils.data.DataLoader:
    image_folder = image_folder
    annotations_file = annotations_file
    cocodataset = CocoDetection(image_folder, annotations_file, resize=resize_shape)
    train_iter = load_data(batch_size, cocodataset)
    return train_iter

def generate_test() -> None:
    train_iter = get_train_iter(image_folder="", annotations_file="")
    batch = next(iter(train_iter))
    images = batch[0][0:50]  
    labels = batch[1][0:50]  

    for image, label in zip(images, labels):
        pil_image = transforms.ToPILImage()(image)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        x, y, h, w = label.tolist()
        opencv_image = cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        x_min, y_min, width, height = label
        cv2.imwrite("{}.jpg".format(generate_random_string(10)), opencv_image)
