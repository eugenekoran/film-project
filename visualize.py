import argparse
import ipdb as pdb
import numpy as np
import torch
from torch.nn.functional import upsample_bilinear
from torch.nn.modules.upsampling import Upsample, UpsamplingBilinear2d
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from utils import tokenize, encode
from preprocess_data import featurize, build_resnet
from scipy.misc import imresize, imread, imsave
from model import FiLMGenerator, FiLMedNet

parser = argparse.ArgumentParser()

parser.add_argument('--image', default='img/CLEVR_test_0000001.png')
parser.add_argument('--question', default='How many objects are there?')
parser.add_argument('--FiLM', default='data/checkpoint.pt')
parser.add_argument('--model', default='resnet101')


def visualize(args):
    #Extract image features
    model = build_resnet(args)
    original_img = imread(args.image, mode='RGB')
    img = imresize(original_img, (224, 224), interp='lanczos')
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    feats = Variable(FloatTensor(featurize([img], model)))

    #Decode question
    checkpoint = torch.load(args.FiLM)
    vocab = checkpoint['vocab']

    tokens = tokenize(args.question)
    encoded = encode(tokens, vocab['q'])
    encoded = Variable(LongTensor(encoded).unsqueeze(0))

    film_generator = FiLMGenerator(len(vocab['q']) + 1) #TODO ############
    filmed_net = FiLMedNet(save_layer=True)

    fg_state = checkpoint['fg_best_state']
    fn_state = checkpoint['fn_best_state']

    film_generator.load_state_dict(fg_state)
    filmed_net.load_state_dict(fn_state)

    film_generator.eval()
    filmed_net.eval()

    film = film_generator(encoded)
    _ = filmed_net(feats, film)

    activations = filmed_net.cf_input
    activations = filmed_net.classifier[0](activations)
    activations = filmed_net.classifier[1](activations)
    activations = filmed_net.classifier[2](activations)

    f_map = (activations ** 2).mean(0).mean(0).sqrt()
    f_map = f_map - f_map.min().expand_as(f_map)
    f_map = f_map / f_map.max().expand_as(f_map)

    pdb.set_trace()
    f_map = (255 * f_map).round()
    upsample = Upsample(size=torch.Size(original_img.shape[:-1]), mode='bilinear')
    channel = upsample(f_map.unsqueeze(0).unsqueeze(0))
    channel = channel.squeeze().unsqueeze(-1).data.numpy()

    filtered_img = np.concatenate((original_img, channel), axis=2)

    imsave('filtered.png', filtered_img)






if __name__ == "__main__":
    args = parser.parse_args()
    visualize(args)
