import torch
from torch import nn
import torchvision

from torch.utils.data import Dataset
import os
from os.path import join
import scipy
from torchvision.datasets.utils import list_dir
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, extract_archive
from torchvision import transforms
import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression

import torch.optim as optim
import torchvision.datasets as datasets

from sklearn.metrics import balanced_accuracy_score
from tqdm import notebook

import warnings
warnings.filterwarnings('ignore')

import sklearn

from tqdm import tqdm_notebook

import os.path
import glob
import sys

from torchvision.datasets.utils import check_integrity
 
from torch.optim.lr_scheduler import LambdaLR

from collections import OrderedDict
from collections import defaultdict

import pickle

from functools import partial

import time
from IPython.display import clear_output


class dogs(Dataset):
    
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    
    
    folder = 'StanfordDogs' # folder with images
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs' # ref from the internet
    

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                        for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images
            
        self.data = [x[0] for x in self._flat_breed_images]
        self.targets = [x[1] for x in self._flat_breed_images]

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]




    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts
    
    
def transform_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def transform_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def load_datasets(set_name, configs, data_transforms, input_size=224):

    if set_name == 'stanford_dogs': 

        train_dataset = dogs(root=configs,
                                 train=True,
                                 cropped=False,
                                 transform= data_transforms['train'],
                                 download=True)        
             
        test_dataset = dogs(root=configs,
                                train=False,
                                cropped=False,
                                transform= data_transforms['test'],
                                download=True)

        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    return train_dataset, test_dataset, classes


from collections import defaultdict

def RandomUnderSample(arr, sampling_rate=0.15, random_state=None):
    x = np.bincount(arr)
    minor_class = x[x != 0].min()
    sample_size = int(sampling_rate * len(arr) // len(set(arr)))
    minor_class, sample_size
    sample_size = minor_class if minor_class < sample_size else sample_size

    dct = defaultdict(list)
    targets = np.array(arr)
    idx = np.arange(len(targets))

    np.random.seed(random_state)
    np.random.shuffle(idx)
    np.random.seed()

    for i, t in zip(idx, targets[idx]):
        dct[t].append(i)

    idx = [i for t, idx in dct.items() for i in idx[:sample_size]]
    return idx


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')
    

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False):
        
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        
        split = 'trainval' if train else 'test'
        
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
            
            
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))
        

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        self.targets = targets
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images
        
        
def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha < 0.5:
        factor = 1.0
    elif alpha < 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


class Resnet50Fc(nn.Module):
    def __init__(self):
        super(Resnet50Fc, self).__init__()
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features # 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Net(nn.Module):
    def __init__(self,num_features, num_classes=100, methods='simple'):
        super(Net,self).__init__()
        self.model_fc = Resnet50Fc() # encoder
        
        self.classifier_layer = nn.Linear(num_features, num_classes)
        
        # ??
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
        self.methods = methods
        
    def forward(self,x):
        
        BSS = 0
        feature = self.model_fc(x)
        
        u,s,v = torch.svd(feature.t())
        
        
        ll = s.size(0) # how many singular values
        
        if self.methods == 'l2+bss':
            for i in range(1):
                BSS = BSS + torch.pow(s[ll-1-i],2)
        else:
            BSS = 0
        outC = self.classifier_layer(feature)
        return(outC, BSS)
    
    
class L2_SP(nn.Module):
    def __init__(self,num_features, num_classes=100):
        super(L2_SP,self).__init__()
        self.model_fc = Resnet50Fc() # encoder
        
        self.classifier_layer = nn.Linear(num_features, num_classes)
        
        # ??
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
        
    def forward(self,x):
        
        feature = self.model_fc(x)
        outC = self.classifier_layer(feature)
        return outC
    
    
class LiveConv2dSliceInfiniteSampler:
  
    def __init__(self, module, batch_size=None):
        assert isinstance(module, torch.nn.Conv2d)
        self.module = module
        self.batch_size = batch_size
        self.shape = self.module.weight.shape[-2:]
        # self.shape is equal to shape of filter in a layer
        # for instance, self.shape = torch.tensor([3,3])
        # if kernel_size of the layer equals 3

    def __str__(self):
        return f"module is {self.module} and batch_size is {self.batch_size}"
  
    # namely it is used for object of class
    def __repr__(self):
        text = f" module = {self.module} , batch_size = {self.batch_size}"
        return type(self).__name__ + "(" + text + ")" 

    def __len__(self):
        return self.module.weight.shape[:-2].numel()
  
    def __iter__(self):
        n_batches = (self.__len__() + self.batch_size - 1)//(self.batch_size)
        while True:
            sequence = torch.randperm(self.__len__(), device = self.module.weight.device) # the second parameter is cuda
            # before for to avoid intersections of batches
            for i in range(n_batches):
                view = self.module.weight.view(-1, *self.shape)
                yield view[sequence[i*self.batch_size : (i+1)*self.batch_size]] 
                
class Discriminator_convolutional(torch.nn.Module):

  def __init__(self , input_dim, c_out_1 , c_out_2):

    """
    Parameters:
     - input_dim : the size of filter (3 or 7)
     - c_out_1 , c_out_2 are optional
     - kernels: 2x2 for 3x3 and 3x3 for 7x7
     - as x = torch.tensor([batch_size, n,n])
    """

    super(Discriminator_convolutional,self).__init__()
     
    
    if not isinstance(c_out_1,int):
      raise ValueError("the parameter c_out should be int, but you've done {}".format(type(c_out_1)))

    if not(input_dim in [5,7]):
      raise ValueError("In case of resnet50, this discriminator supportsonly filters 3x3 and 7x7, filters 1x1 only MLP discriminator. You've taken {}".format(input_dim))

    self.input_dim = input_dim
    self.c_out_1 = c_out_1
    self.c_out_2 = c_out_2

    self.kernel_1 = self.kernel_2 = 3 if self.input_dim == 7 else 2

    self.dim_fc = 3 if self.input_dim == 7 else 3

    self.discr_conv = torch.nn.Sequential(OrderedDict([
        ("conv_discr_1" , torch.nn.Conv2d(1, c_out_1, self.kernel_1)),
        ("bn_discr_1"   , torch.nn.BatchNorm2d(c_out_1) ),
        ("relu_discr_1" , torch.nn.ReLU()),
        ("conv_discr_2" , torch.nn.Conv2d(c_out_1, c_out_2, self.kernel_2)),
        ("bn_discr_2"   , torch.nn.BatchNorm2d(c_out_2)),
        ("relu_discr_2" , torch.nn.ReLU())
    ]))
  
    self.discr_fc = torch.nn.Sequential(OrderedDict([
       ("flatten", torch.nn.Flatten(start_dim = 1, end_dim = -1)),
       ("fcs_discr_1", torch.nn.Linear(self.c_out_2 * self.dim_fc ** 2,1))
    ]))

  def forward(self, x):

    if len(x.shape) != 3:
      raise ValueError("The shape of input tensor should be 3d, but you have {}".format(len(input._shape)))

    x = x.unsqueeze(dim = 1)
    x = self.discr_conv(x)
    
    x = self.discr_fc(x)
    return x


class Discriminator(torch.nn.Module):
    def __new__(cls, input_dim, hidden):
        assert isinstance(hidden, list)

        model = [torch.nn.Flatten()]
        #model.pop()


        hidden_layers = [input_dim] + hidden + [1]
        for in_features, out_features in zip(hidden_layers, hidden_layers[1:]):
            model.extend([
                torch.nn.Linear(in_features, out_features),
                torch.nn.ReLU()
            ])
        model.pop()

        return torch.nn.Sequential(*model)   
    

def dis_loss(discriminator,  real, fake, ell=torch.nn.BCEWithLogitsLoss()):
    out_real = discriminator(real)
    out_fake = discriminator(fake) #(fake.detach())

    loss_real = ell(out_real, torch.full_like(out_real, 0.9))
    loss_fake = ell(out_fake, torch.full_like(out_fake, 0.))
    return loss_real * 0.5 + loss_fake * 0.5

def gen_loss_dis(discriminator, *, fake):
    fake = fake.view(-1, fake.shape[-1], fake.shape[-1])
    return - discriminator(fake).mean()

def gen_loss_task(model, *, input, target, criterion=torch.nn.CrossEntropyLoss()):
    return criterion(model(input), target)



def maker_discriminators_both(flag_convo,
                              hidden_discr_sizes,
                              model,
                              c_out_1 ,
                              c_out_2):

  """
  This function makes convolutional discriminators for 3x3, 7x7 and MLP for 1x1 if flag_convo = True
  Otherwise this function makes MLP discriminators for all filters

  Parameters:
   - flag_convo: is there convolutional_discr
   - hidden_discr_sizes : list of hidden list for mlp 1x1 , 3x3 , 7x7 (HIDDEN_DISCR_NAMES)
   - model: discriminators of layers of which model
  """
  own_dict = {"1":0,"3":1,"7":2}
  discriminators = {}

  if flag_convo == False:
    for name, module in model.named_modules():
      if isinstance(module, torch.nn.Conv2d) and not('downsample' in name):
        discriminators[name] = Discriminator(module.weight.shape[-2:].numel(),hidden_discr_sizes[own_dict[str(module.kernel_size[-1])]]).cpu()
  else:
    for name, module in model.named_modules():
      if isinstance(module,torch.nn.Conv2d) and not('downsample' in name):
        if module.kernel_size[-1] != 1:
          discriminators[name] = Discriminator_convolutional(module.kernel_size[-1], c_out_1, c_out_2).cpu()
        else:
          discriminators[name] = Discriminator(module.weight.shape[-2:].numel(),hidden_discr_sizes[own_dict[str(module.kernel_size[-1])]] ).cpu()

  return discriminators




# auxiliary function for calculation
# If loss will be nan or inf
# then there will be exception 
def protect(scalar):
    """Raise if a scalar is either NaN, or Inf."""
    if not np.isfinite(float(scalar)):
        raise FloatingPointError

    return scalar

# this function calculate loss of discrimiantor 
# and backpropogate gradients through all layers of Discriminator
def make_step(diss_loss_fn, discriminator, discriminator_optimizer, real, fake):
    """
    dis_loss_fn : the loss of discriminator, namely "dis_loss" (see above)
    real : batch of SKD filters in a layer
    fake : batch of current convolutional filters in the same layer
    """
    discriminator_optimizer.zero_grad()
    discriminator_loss = diss_loss_fn(discriminator, real, fake)
    protect(discriminator_loss).backward()
    discriminator_optimizer.step()
    return discriminator_loss.item()

def train_epoch_discriminator(discriminator,
                              loader_skd,
                              diss_loss_fn,
                              discriminator_optimizer,
                              ker_iter_gen,
                              current_steps,
                              critic_steps):
  
    """
    loader_skd :  DataLoader for SKD filters
    ker_iter_gen: generator of a batch of convolutional filters
    critic_steps : amount of established critic's steps  
    current_steps : amount of current critic's steps
    """
  
    #discriminator.requires_grad_(True)
    probs_true, probs_fake, discrim_loss = list(),list(),list()
    # probs_true : probability of discrimiantor on a SKD filter
    # probs_fake : probability of discriminator on a convo filters

    for batch_skd in loader_skd:
        current_steps += 1
        batch_fake = next(ker_iter_gen).detach()

        # to align sizes of batch , when SKD_BATCH_SIZE != BATCH_FAKE_SIZE
        if batch_fake.shape[0] > batch_skd.shape[0]:
            batch_fake = batch_fake[:batch_skd.shape[0]]
        elif batch_fake.shape[0] < batch_skd.shape[0]:
            batch_skd = batch_skd[:batch_fake.shape[0]]

        # make step for a batch
        dis_loss = make_step(diss_loss_fn, discriminator, discriminator_optimizer, batch_skd, batch_fake)

        # calculate probabilities
        with torch.no_grad():
            probs_true.append(discriminator(batch_skd).sigmoid().mean().cpu().numpy())
            probs_fake.append(discriminator(batch_fake).sigmoid().mean().cpu().numpy())
            discrim_loss.append(dis_loss)

        # to check , that current_step is still less, than established critic_steps 
        if current_steps == critic_steps:
            break

    return {"discriminator_loss": discrim_loss,
            "probabilities_SKD" : probs_true,
            "probabilities_fitness": probs_fake},current_steps



def train_discriminator(num_epochs,
                        critic_steps,
                        discriminator,
                        loader_skd,
                        diss_loss_fn,
                        discriminator_optimizer,
                        ker_iter_gen):
  
  
    """
    num_epochs : num epochs of a discriminator
    diss_loss_fn :  the loss of discriminator, namely "dis_loss" (see above)
    loader_skd :  DataLoader for SKD filters
    ker_iter_gen: generator of a batch of convolutional filters
    critic_steps : amount of established critic's steps  
    current_steps : amount of current critic's steps
    """
    discriminator.requires_grad_(True)

    result_dict = { "discriminator_loss": [], "probabilities_SKD":[], "probabilities_fitness":[] }
    current_steps = 0
    for epoch in tqdm_notebook(range(num_epochs)):
        loss_prob_dict, current_steps = train_epoch_discriminator(discriminator, loader_skd, diss_loss_fn,
                                                                  discriminator_optimizer, ker_iter_gen,
                                                                  current_steps, critic_steps)

        for name in result_dict.keys():
            result_dict[name].extend(loss_prob_dict[name])

        # to check , that current_step is still less, than established critic_steps 
        if current_steps >= critic_steps:
            break
    
    discriminator.zero_grad()
    discriminator.requires_grad_(False)

    return result_dict



def plot_learning_curves(history, betas_history, epoch):
    '''
     
    This function is aimed to show up at all loss and accuracy (metrics) during the training of the model

    :param history: (dict)
        accuracy and loss on the training and validation
    '''
    # sns.set_style(style='whitegrid')

    fig = plt.figure(figsize=(50, 30))

    plt.subplot(2,2,1)
    plt.title('Loss', fontsize=15)

    plt.plot(history['loss_full']['train'], label='train_full')
    plt.plot(history['loss_task']['train'], label='loss_task')
    plt.plot(history['loss_gen']['train'], label = 'gen_loss')
    plt.plot(history['loss']['val'],label='val_loss')

    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Num epochs', fontsize=15)
    plt.legend()

    plt.subplot(2,2,2)
    plt.title('Accuracy', fontsize=15)
    plt.plot(history['acc']['train'], label='train')
    plt.plot(history['acc']['val_bal'], label='validation_balanced')
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('Num epochs', fontsize=15)
    plt.legend()


    keys_beta = list(betas_history.keys())
    plt.subplot(2,2,3)
    plt.title('Betas history',fontsize=15)
    for key in keys_beta[:15]:
      plt.plot(betas_history[key],label='{}'.format(key))
    plt.xlabel('Num epochs', fontsize=15)
    plt.ylabel('Beta control', fontsize=15)
    plt.legend()
 
 
    plt.subplot(2,2,4)
    plt.title('Discriminators loss',fontsize=15)
    for key in keys_beta[:15]:
      plt.plot( history['dis_loss'][key],label='discriminator of {}'.format(key))
    plt.xlabel('Num epochs', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.legend()





    #plt.savefig("/content/gdrive/MyDrive/DELTA_Adversarial/{}_adversarial_plot.png".format(epoch))
    plt.show()

def train_CNN(num_epochs,
              max_gen_epochs,
              generator_steps,
              max_gen_draws,
              model,
              discriminators,
              task_loaders,
              beta_dictionary,
              optimizer_generator,
              optimizer_classif,
              optimizer_discriminator,
              discriminator_steps,
              train_loaders_SKD,
              ker_iters_gen,device, epoch_unfreezing):

    """
    num_epochs : num_epochs of CNN
    max_gen_epochs : amount epochs of generator (by default 1)
    generator_steps: amount steps of pseudo-generator
    max_gen_draws : 
    model : CNN
    discriminators : dictionary of discriminators
    task_loaders : dictionary of train loaders of CIFAR10 for CNN
    beta_dictionary : dictionary of constants of pseudo-generator loss
    optimizer_generator: optimizer of CNN
    optimizer_discriminator : optimizer of a discriminator
    discriminator_steps : amount steps of discriminator
    ker_iters_gen: generator for a batch of current convolutional filters in a layer
    """
    # history for interface
    klyuchi = list(discriminators.keys())
    history = defaultdict(lambda: defaultdict(list))

  
    # dictionary for regularization coeffs for each layer
    betas_history = {layer: [value] for layer, value in beta_dictionary.items()}


    accuracy_train = []
    accuracy_val = []
    history_gen_loss = []
    history_discr_loss = []

    for epoch in tqdm_notebook(range(num_epochs)):
    
        optimizer_classif.param_groups[0]['lr'] = learning_rate_schedule(0.1, epoch+1, num_epochs)

        train_loss   =    0
        train_acc    =    0
        loss_task_sum =   0
        gen_loss_task_sum = 0

        val_loss =        0
        val_acc  =        0 

        start_time = time.time()

        # training of model

        # turn off disriminators
        for name, dis in discriminators.items():
            dis.to(device)
            dis.requires_grad_(False)

        model.train()

        # model on cuda
        model.to(device)

        # which is epoch
        if epoch >= epoch_unfreezing:
          # classif and encoder
            model.requires_grad_(True) 
        else:
          # classif  only
            model.requires_grad_(False)
            model.classifier.requires_grad_(True)


        for generator_epoch in range(max_gen_epochs):
            for num_batch,(batch_x,batch_y) in zip(range(generator_steps), task_loaders['train']):

                logits = model(batch_x.to(device)) #

                optimizer_generator.zero_grad()
                optimizer_classif.zero_grad()

                loss_task = torch.nn.CrossEntropyLoss()(logits, batch_y.to(device).long()) 

                if epoch >= epoch_unfreezing:
                    # task_loss for encoder+classif
                    loss_task.backward(retain_graph = True) # task loss on cuda
                else:
                    # for classifier only cross entropy
                    loss_task.backward()

                loss_task = loss_task.item()

                y_pred_train = logits.argmax(1).cpu()#.numpy()
                #loss_task = torch.nn.CrossEntropyLoss()(logits.cpu(), batch_y.cpu().long()) 

                # logits on cuda lets remove
                del logits
                train_acc += (batch_y == y_pred_train).float().mean()

                #loss_task = gen_loss_task(model, input=batch_x, target = batch_y.long())

                # calculate pseudo-generator loss throughout all layers

                if epoch >= epoch_unfreezing:
                  # calculate generator loss if convo layers are not freezed
                    dis_terms = {
                        (layer): gen_loss_dis(dis, fake=next(ker_iters_gen[layer]))
                         for layer, dis in discriminators.items()
                         }


                  # calculate sum loss : CrossEntropyLoss + pseudo-generator loss of each layer
                    value = sum(
                    beta_dictionary.get(layer) * term
                    for layer, term in dis_terms.items())

                    gen_loss = value.item()
                    print("generative_loss = {}".format(gen_loss))
                     
                    value.backward()
                    del value 

                if epoch >= epoch_unfreezing:
                    optimizer_generator.step()
                    optimizer_classif.step()

                    train_loss += gen_loss + loss_task
                    loss_task_sum += loss_task
                    gen_loss_task_sum += gen_loss

                else:
                    optimizer_classif.step()

                    train_loss += loss_task
                    loss_task_sum += loss_task  - 0.1

                # until without scheduler
                """
                optimizer_generator.step()
                scheduler.step()
                """
                #model.cpu()

                # to get accuracy let's calculate predictions of the CNN

                 
                #loss_task_sum += np.sum(loss_task.detach().cpu().numpy())
                #gen_loss_task_sum += np.sum(value.detach().cpu().numpy())



        if epoch >= epoch_unfreezing:
            history_gen_loss.append(gen_loss)
        # normalizing train_loss and train_acc  
        train_loss /= len(task_loaders['train'])
        loss_task_sum /= len(task_loaders['train'])

        if epoch >= epoch_unfreezing:
          gen_loss_task_sum /= len(task_loaders['train'])
          history['loss_gen']['train'].append(gen_loss_task_sum)
        else:
          history['loss_gen']['train'].append(train_loss - loss_task_sum)

        train_acc /= len(task_loaders['train'])
        accuracy_train.append(train_acc)

        history['loss_full']['train'].append(train_loss)
        history['loss_task']['train'].append(loss_task_sum)

        history['acc']['train'].append(train_acc)
          




        # training of discriminator
        # model's gradient doesn't backprop

        # model on cpu
        #model.cpu()
        #model.zero_grad() !!!
        if epoch >= epoch_unfreezing:
          # discriminators on cuda
          #for name, dis in discriminators.items():
          #  dis.to(device)
            for discriminator in discriminators.values():
                discriminator.requires_grad_(True)
            
             
            for _ in range(discriminator_steps):
                loss = {}
                for layer, dis in discriminators.items(): 

                # get a batch of SKD filters and a batch of current convo filters
                    real = next(iter(train_loaders_SKD[layer])).to(device)
                    fake = next(ker_iters_gen[layer]).detach()

                # to align batch sizes when BATCH_SIZE_SKD != BATCH_SIZE_CONVO
                    if real.shape[0] < fake.shape[0]:
                        fake = fake[:real.shape[0]]
                    elif real.shape[0] > fake.shape[0]:
                        real = real[:fake.shape[0]]

                # calculate loss of dsicriminator in a batch
                    loss[layer] = dis_loss(dis, real=real, fake=fake)

                # to save which probability for fake and true filter pera batch that are true 
                    with torch.no_grad():
 

                        #real = real.reshape(1,real.shape[0])
                        #fake = fake.reshape(1,fake.shape[0])
                        history['prob_true'][layer].append(dis(real).sigmoid().mean().item())
                        history['prob_fake'][layer].append(dis(fake).sigmoid().mean().item())


                for name in klyuchi:
                  history['dis_loss'][name].append(loss[name].item())

                value = sum(loss.values())
                print("discriminators loss {}".format(value))
                


              #backprop for a discriminator of a layer
                for optimizer in optimizer_discriminator.values():
                    optimizer.zero_grad()

                protect(value).backward()
                for optimizer in optimizer_discriminator.values():
                    optimizer.step()
        

             

        if epoch > epoch_unfreezing:
            history_discr_loss.append(value)
        # discriminators on cpu
        for n,dis in discriminators.items():
            dis.zero_grad()
            #dis.cpu()

        with torch.no_grad():
          # eval model 
          #model.to(device)
            model.eval()

            massiv_logits =[]
            massiv_true = []
            
            for batch_x,batch_y in task_loaders['val']:


                # calculate CrossEntropyLoss
                #loss_ = gen_loss_task(model,input=batch_x,target = batch_y)
                logits_ = model(batch_x.to(device))

                y_pred_ = logits_.max(1)[1].detach().cpu().numpy()

                val_acc += np.mean(batch_y.cpu().numpy() == y_pred_)

                # calculate predictions in a validation
                val_loss += np.sum(torch.nn.CrossEntropyLoss()(logits_, batch_y.to(device).long()).detach().cpu().numpy())
                del logits_
                del batch_x

                massiv_logits.extend(y_pred_)
                massiv_true.extend([ x.item() for x in batch_y.cpu()])


            #model.cpu()
            val_acc_balanced = balanced_accuracy_score(massiv_true, massiv_logits)
            #  normalizing of loss and accuracy
            val_loss /= len(task_loaders['val'])
            val_acc /= len(task_loaders['val']) 

            history['loss']['val'].append(val_loss)
            history['acc']['val_bal'].append(val_acc_balanced)
            history['acc']['val_unbal'].append(val_acc)
             
        # parameter of regularization
        if epoch >= epoch_unfreezing:
        ## feedback for the generator's objective hyperparameters
            out = {}  # inflate, keep, or deflate the weight in the generator's objective
            for layer, value in beta_dictionary.items():
                p_real, p_fake = history['prob_true'][layer][-1], history['prob_fake'][layer][-1]
                if p_fake < 0.2:
                    value = value * 2.

                elif p_fake > 0.55:
                    value = value / 2.

                out[layer] = max(1e-3, min(1e3, value))

                betas_history[layer].append(out[layer])

            beta_dictionary = out 

        clear_output()

        #print results after each epoch
        print("Epoch {} of {} took {:.3f}s".format(
                  epoch + 1, num_epochs, time.time() - start_time))
        #print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        #print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        #print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc_balanced * 100))

        plot_learning_curves(history, betas_history, epoch)
    return model, history, betas_history,history_gen_loss, history_discr_loss

def get_data(data_dir, data_transforms):
     
    train_dataset = torchvision.datasets.ImageFolder(data_dir ,transform=data_transforms['train'])
    val_dataset_norm = torchvision.datasets.ImageFolder(data_dir, transform = data_transforms['val'])
    
    classes = train_dataset.classes

 

    
    
    return train_dataset, val_dataset_norm, classes


def reg_classifier(model, fc_name , device):
  l2_cls = torch.tensor(0.).to(device)
  for name, param in model.named_parameters():
    if name.startswith(fc_name):
        l2_cls += 0.5 * torch.norm(param) ** 2
  return l2_cls

# !! Important: it doesn't change shape of fea tensor
def flatten_outputs(fea):
  return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))

    # regularization with attention mechanism
def reg_att_fea_map(inputs, model_source, device ,
                    layer_outputs_source, layer_outputs_target, 
                    channel_weights):
  _ = model_source(inputs)
  fea_loss = torch.tensor(0.).to(device)
  for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
            b, c, h, w = fm_src.shape
            fm_src = flatten_outputs(fm_src)
            fm_tgt = flatten_outputs(fm_tgt)
            div_norm = h * w

            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)

            distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)

            fea_loss += 0.5 * torch.sum(distance)

  return fea_loss

def plot_delta_plots(model_source, model_target, history, epoch):


  fig = plt.figure(figsize=(10,10))

  plt.subplot(3,1,1)
  plt.plot(history['train']['loss'], label = 'full_loss')
  plt.plot(history['train']['loss_task'], label=  'task_loss')
  plt.plot(history['train']['loss_reg'],  label=  'reg_loss' )
  plt.plot(history['train']['loss_fea']      ,  label = 'loss_feature' )
  plt.xlabel('num_epochs')
  plt.ylabel('loss')
  plt.legend()

  plt.subplot(3,1,2)
  plt.plot(history['train']['acc'],  label='accuracy train unbalanced' )
  plt.plot(history['val']['acc'],label = 'val_acc balanced')
  plt.xlabel('num_epochs')
  plt.ylabel('accuracy')
  plt.legend()

  #plt.savefig("/content/gdrive/MyDrive/DELTA_TL/plot_delta_epoch_{}".format(epoch))

  plt.show()
