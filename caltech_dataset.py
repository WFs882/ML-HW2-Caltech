from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(directory, class_to_idx):
    instances = []
    dataset = []
    directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys(),key=str.lower):
        #print(target_class)
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(root, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                head, tail = os.path.split(root)
                path = os.path.join(tail, fname)
                instances.append(item)

        num_of_instances=len(instances)

        i=0
        next=True
        while(i<num_of_instances):
            if(next is True):
                next = False
                currentImage = file.readline()
                currentImage = currentImage.replace("/", "\\").replace("\n","")
            if(currentImage.find("Google")!=-1):
                next = True
            elif(currentImage == instances[i][0]):
                dataset.append(instances[i])
                next = True
                i=i+1
            else:
                i=i+1
   return dataset         
    
class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        classes = [d.name for d in os.scandir(root) if (d.is_dir() and d.name!="BACKGROUND_Google")]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        '''
        instances = []
        dataset = []

        filePath = "Caltech101/" + split + ".txt"
        file = open(filePath, "r")
        number_of_lines = len(file.readlines())
        #print(number_of_lines)
        file = open(filePath, "r")

        for target_class in sorted(class_to_idx.keys(),key=str.lower):
            #print(target_class)
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    head, tail = os.path.split(root)
                    path = os.path.join(tail, fname)
                    item = path, class_index
                    instances.append(item)

        num_of_instances=len(instances)

        i=0
        next=True
        while(i<num_of_instances):
            if(next is True):
                next = False
                currentImage = file.readline()
                currentImage = currentImage.replace("/", "\\").replace("\n","")
            if(currentImage.find("Google")!=-1):
                next = True
            elif(currentImage == instances[i][0]):
                dataset.append(instances[i])
                next = True
                i=i+1
            else:
                i=i+1
        '''
        samples = make_dataset(self.root, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.target = [s[1] for s in samples]

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        #image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int
        
        path, label = self.samples[index]
        path = os.path.join(dir, path)
        
        image = pil_loader(path)
        
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(target)
            
        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        return len(self.samples) # Provide a way to get the length (number of elements) of the dataset
