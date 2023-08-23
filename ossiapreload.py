import torch                                                       
import torchaudio
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn import functional as F
from torch import nn, optim
Resample = torchaudio.transforms.Resample(44100, 48000, resampling_method='kaiser_window')
device = 'cuda:0'
Resample = Resample.to(device)

from pathlib import Path
import random
import numpy as np
from scipy import interpolate as sp_interpolate
import librosa


sampling_rate = 44100
sr = sampling_rate

hop_length = 128

segment_length = 1024
n_units = 2048
latent_dim = 256
device = 'cuda:0'

batch_size = 256

audio_fold = Path(r'./content/2022-zkm-workshop/ltsp/erokia/audio')
audio = audio_fold
lts_audio_files = [f for f in audio_fold.glob('*.wav')]

# Models 

class raw_VAE(nn.Module):
  def __init__(self, segment_length, n_units, latent_dim):
    super(raw_VAE, self).__init__()

    self.segment_length = segment_length
    self.n_units = n_units
    self.latent_dim = latent_dim
    
    self.fc1 = nn.Linear(segment_length, n_units)
    self.fc21 = nn.Linear(n_units, latent_dim)
    self.fc22 = nn.Linear(n_units, latent_dim)
    self.fc3 = nn.Linear(latent_dim, n_units)
    self.fc4 = nn.Linear(n_units, segment_length)

  def encode(self, x):
      h1 = F.relu(self.fc1(x))
      return self.fc21(h1), self.fc22(h1)

  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std

  def decode(self, z):
      h3 = F.relu(self.fc3(z))
      return F.tanh(self.fc4(h3))

  def forward(self, x):
      mu, logvar = self.encode(x.view(-1, self.segment_length))
      z = self.reparameterize(mu, logvar)
      return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_beta, segment_length):
  recon_loss = F.mse_loss(recon_x, x.view(-1, segment_length))

  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

  return recon_loss + ( kl_beta * KLD)
  
# Datasets 

class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_np, segment_length, sampling_rate, hop_size, transform=None):
        
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.hop_size = hop_size
        
        if segment_length % hop_size != 0:
            raise ValueError("segment_length {} is not a multiple of hop_size {}".format(segment_length, hop_size))

        if len(audio_np) % hop_size != 0:
            num_zeros = hop_size - (len(audio_np) % hop_size)
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0,0))

        self.audio_np = audio_np
        
    def __getitem__(self, index):
        
        # Take segment
        seg_start = index * self.hop_size
        seg_end = (index * self.hop_size) + self.segment_length
        sample = self.audio_np[ seg_start : seg_end ]
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return (len(self.audio_np) // self.hop_size) - (self.segment_length // self.hop_size) + 1

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)

class TestDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_np, segment_length, sampling_rate, transform=None):
        
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        
        if len(audio_np) % segment_length != 0:
            num_zeros = segment_length - (len(audio_np) % segment_length)
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0,0))

        self.audio_np = audio_np
        
    def __getitem__(self, index):
        
        # Take segment
        seg_start = index * self.segment_length
        seg_end = (index * self.segment_length) + self.segment_length
        sample = self.audio_np[ seg_start : seg_end ]
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.audio_np) // self.segment_length

state = torch.load(Path(r'./content/2022-zkm-workshop/nospectral/erokia/spectralvae/run-000/checkpoints/ckpt_00500'))
model = raw_VAE(segment_length, n_units, latent_dim).to(device)
model.load_state_dict(state['state_dict'])
# model = None # torch.load your model  
if model != None:
    print(model)
    model.eval()

test_audio_1_path = lts_audio_files[random.randint(0, len(lts_audio_files) - 1)]
test_audio_1, fs = librosa.load(test_audio_1_path, sr=None)
test_audio_2_path = lts_audio_files[random.randint(0, len(lts_audio_files)- 1)]
test_audio_2, fs = librosa.load(test_audio_2_path, sr=None)

if test_audio_1.shape[0] < test_audio_2.shape[0]:
    while test_audio_1.shape[0] < test_audio_2.shape[0]:
        test_audio_1 = np.concatenate((test_audio_1, test_audio_1), 0)
    test_audio_1 = test_audio_1[:test_audio_2.shape[0]] 

else:
    while test_audio_2.shape[0] < test_audio_1.shape[0]:
        test_audio_2 = np.concatenate((test_audio_2, test_audio_2), 0)
    test_audio_2 = test_audio_2[:test_audio_1.shape[0]]

# Create the dataset
test_dataset1 = TestDataset(test_audio_1, segment_length = segment_length, sampling_rate = sampling_rate, transform=ToTensor())
test_dataset2 = TestDataset(test_audio_2, segment_length = segment_length, sampling_rate = sampling_rate, transform=ToTensor())

test_dataloader1 = DataLoader(test_dataset1, batch_size = batch_size, shuffle=False)
test_dataloader2 = DataLoader(test_dataset2, batch_size = batch_size, shuffle=False)