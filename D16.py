import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import os 
# from vocabulary import Vocabulary
import json
from PIL import Image
from torchvision import transforms
vocab_threshold=None
vocab_file='./vocab.pkl'
start_word="<start>"
end_word="<end>"
unk_word="<unk>"



class my_vocab:
    def __init__(self, vocab_file):
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_file = vocab_file
        self.build_vocab()

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx) 

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_tags()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    def add_tags(self):
        with open(self.vocab_file,'r') as f:
            for idx,line in enumerate(f):
                try:
                    line = line.strip()
                    img_file, tags_list = line.split("\t\t")
                    tags_list = json.loads(tags_list)
                    for tag in tags_list: self.add_word(tag)
                except Exception as e:
                    print("[ERROR] reading vocabulary file {}".format(self.vocab_file))
                    print("[line {}".format(idx),line)
def get_loader():
    dataset = Dataset('inference')
    data_loader = data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=0)
    return data_loader
          
class Dataset(data.Dataset):
    def __init__(self,mode,vocab='',dataset_file = '/data/shubham/mscoco/captions.txt',max_length=20,cnn_mode=False):
        self.mode=mode
        self.cnn_mode=cnn_mode
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        if  self.mode in ['train','inference']:
            print(self.mode)
            self.vocab = vocab
            self.X=[]
            self.Y=[]
            print("loading data.")
            with open(dataset_file,'r') as f:
                for line in f:
                        line = line.strip()
                        img_file, tags_list = line.split("\t\t")
                        tags_list = json.loads(tags_list)
                        y = []
                        y.append(self.vocab(self.vocab.start_word))
                        y.extend([self.vocab(token) for token in tags_list])
                        y = y[0:max_length+1]
                        while len(y)<=max_length: 
                            y.append(self.vocab(self.vocab.end_word))
                        y.append(self.vocab(self.vocab.end_word))
                        y = torch.Tensor(y).long()
                        self.X.append(img_file)
                        self.Y.append(y)
            self.img_folder = '/data/deva/mscoco/train2014'
            if(self.mode=='inference'):
                self.img_folder = '/data/deva/mscoco/val2014'
            print("loading data done",len(self.X))
        elif (self.mode=='test'):
            print ("Loading Test dataset")
            self.img_folder = '/data/deva/mscoco/test2014'
            self.paths = [files for root,dirs, files in os.walk(self.img_folder)]
            self.paths = self.paths[0]
            self.transform_test = transforms.Compose([ 
                 transforms.Resize(256),                          
                 transforms.CenterCrop(224),                             
                 transforms.ToTensor(),                           
                 transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])
            print("loading data done",len(self.paths))
            self.X=self.paths
    def __len__(self):
        'Denotes the total number of samples'
        if  self.mode in ['train','inference']:
            return len(self.X)
        elif(self.mode=='test'):
            return len(self.paths)
    def __getitem__(self, index):
    
        if  self.mode in ['train','inference']:
             'Generates one sample of data'
             # Select sample
             path = self.X[index]
             
             if (self.mode == 'train'):
                  if self.cnn_mode: 
                      embedding = torch.from_numpy(np.load(os.path.join('/data/shubham/mscoco/image_embeddings_resnet_18_layer4/', path+'.txt.npy'))).float()
                  else:
                      embedding = torch.from_numpy(np.loadtxt(os.path.join('/data/shubham/mscoco/image_embeddings_resnet_18/', path+'.txt'))).float()
             else:
                  if self.cnn_mode:
                      embedding = torch.from_numpy(np.load(os.path.join('/data/shubham/mscoco/image_embeddings_resnet_18_layer4_val/', path+'.txt.npy'))).float()
                  else:
                      embedding = torch.from_numpy(np.loadtxt(os.path.join('/data/shubham/mscoco/image_embeddings_resnet_18/', path+'.txt'))).float()
            
             # Load data and get label
             tags = self.Y[index]
             abs_path = os.path.join(self.img_folder,path)
             image = Image.open(abs_path).convert("RGB")
             image = self.normalize(self.to_tensor(self.scaler(image)))
             if self.mode=='train':
                  return image, tags
             if self.mode=='inference':
                  abs_path = os.path.join(self.img_folder,path)
                  return image,tags, abs_path

        if self.mode=='test':
             path = self.paths[index]
             abs_path = os.path.join(self.img_folder,path)
             PIL_image = Image.open(abs_path).convert('RGB')
             orig_image = np.array(PIL_image)
             image = self.transform_test(PIL_image)
             # return original image and pre-processed image tensor
             return orig_image, image,abs_path

