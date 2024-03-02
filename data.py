import os
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter as Inter
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from PIL import Image
import librosa
import scipy.io as scio
from mindspore import Tensor

def audio_extract(wav_file, sr=16000):
    wav=librosa.load(wav_file,sr=sr)[0]
    # Takes a waveform(length 160000,sampling rate 16000) and extracts filterbank features(size 400*64)
    spec=librosa.core.stft(wav,n_fft=4096,hop_length=200,win_length=1024,window="hann",center=True,pad_mode="constant")
    mel=librosa.feature.mfcc(S=np.abs(spec), sr=sr,n_mfcc=64)#melspectrogram(S=np.abs(spec),sr=sr,n_mels=64,fmax=8000)
    #print(mel.shape)
    logmel=librosa.core.power_to_db(mel[:,:300])#300
    if logmel.shape[1]!=300:
       logmel=np.column_stack([logmel,[[0]*(300-int(logmel.shape[1]))]*64])
    return logmel.T.astype("float32")

import os
import numpy as np
import scipy.io as scio
from PIL import Image
import numpy.random as random

class PrecompDataset:
    def __init__(self, data_split, opt):
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']
        self.aud_path = opt['dataset']['audio_path']
        self.aud_mat_path = opt['dataset']['audio_mat_path']
        self.audios = []
        self.images = []
        if data_split != 'test':
            aud_mat = scio.loadmat(self.aud_mat_path + "train_audios.mat")
            with open(self.loc + '%s_auds_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    aud = aud_mat[str(line.strip())[2:-1]]
                    # Use numpy to expand dimensions instead of MindSpore ops
                    aud = np.expand_dims(aud.astype(np.float32), axis=0)
                    self.audios.append(aud)

            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            aud_mat = scio.loadmat(self.aud_mat_path + "test_audios.mat")
            with open(self.loc + '%s_auds.txt' % data_split, 'rb') as f:
                for line in f:
                    aud = aud_mat[str(line.strip())[2:-1]]
                    # Use numpy to expand dimensions
                    aud = np.expand_dims(aud.astype(np.float32), axis=0)
                    self.audios.append(aud)
            
            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        
        self.length = len(self.audios)
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # Define normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
        
        # Define transformations for training and other splits
        if data_split == "train":
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform

    def train_transform(self, image):

        # Resize
        image = image.resize((278, 278), Image.BILINEAR)
        # Random rotation
        angle = random.randint(-90, 90)
        image = image.rotate(angle)
        # Random crop
        crop_size = (224, 224)
        x = random.randint(0, image.width - crop_size[0])
        y = random.randint(0, image.height - crop_size[1])
        image = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
        # ToTensor
        image = np.array(image, dtype=np.float32) / np.float32(255.0)
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
           
        return image

    def test_transform(self, image):
        # Resize
        image = image.resize((224, 224), Image.BILINEAR)
        # ToTensor
        image = np.array(image, dtype=np.float32) / np.float32(255.0)
        # Normalize
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, index):
        img_id = index // self.im_div
        audio = self.audios[index]
        image_path = os.path.join(self.img_path, str(self.images[img_id])[2:-1])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image.copy(), audio.copy()

    def __len__(self):
        return self.length

def get_precomp_loader(data_split, batch_size=100, shuffle=True, opt={}):
    dset = PrecompDataset(data_split, opt)

    data_loader = ds.GeneratorDataset(dset, ["image", "audio"], shuffle=shuffle, num_parallel_workers=opt['dataset']['workers'])
    data_loader = data_loader.batch(batch_size)
    return data_loader


def get_loaders(opt={}):
    train_loader = get_precomp_loader( 'train',
                                      opt['dataset']['batch_size'], True, opt)
    val_loader = get_precomp_loader( 'val',
                                    opt['dataset']['batch_size_val'], False, opt)
    return train_loader, val_loader


def get_test_loader(opt):
    test_loader = get_precomp_loader( 'test',
                                      opt['dataset']['batch_size_val'], False, opt)
    return test_loader