from collections import defaultdict
import torch
from torch.autograd import Variable

def tokenize(s, keep_punct=[';', ','], remove_punct=['?', '.']):
    '''
    Tokenize a sequence, convert a string into a list of tokens, add start and
    end tokens. Can keep and remove specified punctuation marks
    '''
    for p in keep_punct:
        s = s.replace(p, ' {}'.format(p))
    for p in remove_punct:
        s = s.replace(p, '')

    tokens = s.split()
    tokens.insert(0, '<START>')
    tokens.append('<END>')
    return tokens

def create_vocab(questions,
                min_count=1,
                keep_punct=[';', ','],
                remove_punct=['?', '.']):
    '''Build a vocabulary from a sequence of questions. Each word is mapped to
    a single index. Special item get hardcoded indices.'''

    count_dict = defaultdict(int)
    for q in questions:
        q_tokens = tokenize(q)
        for token in q_tokens:
            count_dict[token] += 1
    token_vocab = {'<NULL>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    for token, count in sorted(count_dict.items()):
        if token not in token_vocab and count >= min_count:
            token_vocab[token] = len(token_vocab)
    return token_vocab

def encode(tokens, token_vocab):
    '''Encode sequence of tokens with vocabulary'''
    encoded = []
    for token in tokens:
        if token not in token_vocab:
            token='<UNK>'
        encoded.append(token_vocab[token])
    return encoded

def invert_vocab(token_vocab):
    '''Creates inverted vocabulary'''
    return {token_vocab[token]: token for token in token_vocab}

def decode(encoded, idx_vocab):
    decoded = []
    for idx in encoded:
        decoded.append(idx_vocab[idx])
        if idx == 2:
            break
    return ' '.join(decoded)

def create_map(shape, start=-1, end=1):
    '''
    Given shape mxn, returns mxn coordinate map, ranging from -1 to 1 in
    x and y directions respectively.
    '''
    m, n = shape
    xs = torch.linspace(start, end, steps=n).cuda()
    ys = torch.linspace(start, end, steps=m).cuda()
    xs = xs.unsqueeze(0).expand(shape).unsqueeze(0)
    ys = ys.unsqueeze(1).expand(shape).unsqueeze(0)
    coord_map = Variable(torch.cat((xs, ys), dim=0))
    return coord_map
