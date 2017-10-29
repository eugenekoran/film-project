import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import LongTensor
from torch.autograd import Variable
import ipdb as pdb
from utils import create_map

class FiLM(nn.Module):
    def forward(self, x, gammas, betas):
        '''
        Feature-wise linear modulation
        '''
        gammas = gammas.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        betas = betas.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return gammas * x + betas

class FiLMGenerator(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        #Initialize layers
        self.embedding = nn.Embedding(vocab_size, 200)
        self.gru = nn.GRU(200, 4096, 1, batch_first=True)
        self.linear = nn.Linear(4096, 1024)
        #Initialize Convolutonal and Linear modules with He initialization
        init_modules(self.modules())

    def forward(self, x):
        #Get the indices of the last words in each sentence
        N, L = x.size()
        idx = LongTensor(N).fill_(L - 1)
        x_numpy = x.cpu().data.numpy()
        for i in range(N):
            for j in range(L - 1):
                if x_numpy[i, j] != 0 and x_numpy[i, j + 1] == 0:
                    idx[i] = j
                    break
        idx = Variable(idx.type_as(x.data))
        #Create a mask to gather hidden states from the GRU
        mask = idx.view(N, 1, 1).expand(N, 1, 4096)

        #Embed questions
        embeded = self.embedding(x)
        
        #GRU
        h0 = Variable(torch.zeros(1, N, 4096).type_as(embeded.data))
        out, _ = self.gru(embeded, h0)

        #Gather a hidden state at the end of the sentence
        encoded = out.gather(1, mask).view(N, 4096)

        #Make linear transformation and split data into 4 film layers
        film = self.linear(encoded).view(N, 4, 256)
        return film

class FiLMedNet(nn.Module):
    def __init__(self, num_modules=4, save_layer=False):
        super().__init__()
        self.num_modules = num_modules
        self.save_layer = save_layer
        self.cf_input = None

        #Create coordinate map to facilitate spatial reasoning
        self.coord_map = create_map((14, 14))

        #Create stem
        self.stem = nn.Sequential(
            nn.Conv2d(1026, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        #Create FiLMed Network body
        layers = []
        for i in range(num_modules):
            layers.append(ResBlock())
        self.filmed_blocks = nn.Sequential(*layers)

        #Create final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(130, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=14),
            Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32))

    def forward(self, x, film):
        #Split film layer into gammas and betas
        gammas, betas = torch.split(film, 128, dim=-1)

        #Expand coordinate maps to match batch size. Concat them to feature maps
        batch_maps = self.coord_map.unsqueeze(0).expand(x.size(0),
                                                        *self.coord_map.size())
        x = torch.cat((x, batch_maps), dim=1)

        #Stem
        x = self.stem(x)

        #Residual Blocks
        for i, block in enumerate(self.filmed_blocks):
            x = block(x, gammas[:, i], betas[:, i], batch_maps)

        #Concatenate coordinate maps before final classifier
        x = torch.cat((x, batch_maps), dim=1)

        #If save_layer, save activations before classifier module
        if self.save_layer:
            self.cf_input = x

        out = self.classifier(x)
        return out

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(130, 128, kernel_size=1)
        self.conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(128, affine=False)
        self.film = FiLM()
        #Initialize Convolutonal and Linear modules with He initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform(m.weight)

    def forward(self, x, gammas, betas, batch_maps):

        #Projection
        x = torch.cat((x, batch_maps), dim=1)
        x = F.relu(self.input_projection(x))
        residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = self.film(x, gammas, betas)
        x = F.relu(x)
        out = x + residual
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init_modules(modules):
    '''
    Initialize Convolutonal and Linear modules with He initialization
    '''
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_uniform(m.weight)
