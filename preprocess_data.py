import os
import argparse
import json
import h5py
from tqdm import tqdm
import numpy as np
from scipy.misc import imread, imresize
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn import Sequential
import torchvision
import ipdb as pdb

from utils import tokenize, create_vocab, encode

'''
Example usage:
python preprocess_data.py \
 --questions_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
 --save_questions_h5_to data/train_questions.h5 \
 --save_vocab_to data/vocab.json \
 --image_folder data/CLEVR_v1.0/images/train \
 --save_features_h5_to data/train_features.h5 \
'''

parser = argparse.ArgumentParser()

#Question preprocessing arguments:
parser.add_argument('--questions_json', required=True)
parser.add_argument('--save_questions_h5_to', required=True)
parser.add_argument('--read_vocab', default=None) #read saved vocab for val ans test data
parser.add_argument('--save_vocab_to', default='data/vocab.json')

#Image preprocessing arguments:
parser.add_argument('--image_folder', required=True)
parser.add_argument('--save_features_h5_to', required=True)
parser.add_argument('--model', default='resnet101',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])


def preprocess_questions(args):
    '''Preprocess questions and save them to h5 file'''
    #Open questions json file
    with open(args.questions_json, 'r') as f:
        questions = json.load(f)['questions']

    #If there is a vocabulary, load it. If no, create new one.
    if args.read_vocab:
        print ('Loading vocabulary from {}'.format(args.read_vocab))
        with open(args.read_vocab, 'r') as f:
            vocab = json.load(f)
    else:
        print ('Building vocabulary')
        question_token_vocab = create_vocab((q['question'] for q in questions))
        answer_token_vocab = create_vocab((q['answer'] for q in questions))
        vocab={'q': question_token_vocab, 'a': answer_token_vocab}
        with open(args.save_vocab_to, 'w') as f:
            json.dump(vocab, f)

    #Encode questions and answers
    print ('Encoding questions')
    questions_encoded = []
    idxs = []
    image_idxs = []
    answers = []

    for idx, q in enumerate(questions):
        idxs.append(idx)
        image_idxs.append(q['image_index'])

        q_tokenized = tokenize(q['question'])
        q_encoded = encode(q_tokenized, vocab['q'])
        questions_encoded.append(q_encoded)
        if 'answer' in q:
            answers.append(vocab['a'][q['answer']])

    #Pad questions with <NULL>`s to make all questions have the same lenght
    max_length = max(len(q) for q in questions_encoded)
    for q in questions_encoded:
        while len(q) < max_length:
            q.append(0)

    #Save questions to h5py file
    with h5py.File(args.save_questions_h5_to, 'w') as f:
        f.create_dataset('idxs', data=np.asarray(idxs))
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('questions', data=np.asarray(questions_encoded))
        if answers:
            f.create_dataset('answers', data=np.asarray(answers))
    return

def preprorcess_images(args):
    '''
    Extract featurs from images and save them in specified h5 file.
    '''
    #Save paths to images
    paths = os.listdir(args.image_folder)
    paths.sort()

    #Build a model
    print ('Building ResNet feature extractor ...')
    model = build_resnet(args)
    #Save features to output_h5_file
    with h5py.File(args.save_features_h5_to, 'w') as f:

        feat_dataset = f.create_dataset('features',
                                        (len(paths), 1024, 14, 14),
                                        dtype=np.float32)
        i0 = 0
        batch = []

        print ('Loading features ...')

        for path in tqdm(paths):
            img = imread(args.image_folder + '/' + path, mode='RGB')
            img = imresize(img, (224, 224), interp='lanczos')
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            batch.append(img)
            #If there are 128 images in the batch load them into the dataset
            if len(batch) == 128:
                features = featurize(batch, model)
                i1 = i0 + len(batch)
                feat_dataset[i0:i1] = features
                i0 = i1
                batch = []
        # Load remaining images to the dataset
        if len(batch) > 0:
            features = featurize(batch, model)
            i1 = i0 + len(batch)
            feat_dataset[i0:i1] = features
    return

def featurize(batch, model):
    '''
    Run batch of images through model and return features
    input:
        batch: list of np.ndarrays
        model: model
    output:
        feats: np.ndarray
    '''
        #Use mean and std of ImageNet
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    #Create batch and wrap it into Variable
    batch = np.concatenate(batch, 0).astype(np.float32)
    batch = (batch / 255.0 - mean) / std
    batch = FloatTensor(batch).cuda()
    batch = Variable(batch, volatile=True)

    #Get features
    features = model(batch)
    features = features.data.cpu().clone().numpy()
    return features

def build_resnet(args):
    '''
    Build ResNet. Return pretrained model
    '''
    #Get pretrained ResNet
    cnn = getattr(torchvision.models, args.model)(pretrained=True)

    #Create list of layers. Take layers up to third.
    layers = [cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1, cnn.layer2,
                                                                    cnn.layer3]
    #Create model
    model = Sequential(*layers)
    model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess_questions(args)
    preprorcess_images(args)
