import argparse
import json
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable
import ipdb as pdb

from dataloader import MyDataLoader
from model import FiLMGenerator, FiLMedNet

parser = argparse.ArgumentParser()

parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--train_questions_h5', default='data/train_questions.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--val_questions_h5', default='data/val_questions.h5')
parser.add_argument('--vocab', default='data/vocab.json')
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--start_from_checkpoint', default=None)
parser.add_argument('--save_checkpoint_to', default='data/checkpoint.pt')

def train(args):
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab['q'])

    with MyDataLoader(args.train_features_h5,
                      args.train_questions_h5) as train_loader, \
         MyDataLoader(args.val_features_h5,
                      args.val_questions_h5) as val_loader:

        film_generator = FiLMGenerator(vocab_size)
        filmed_net = FiLMedNet()

        if args.start_from_checkpoint:
            print ('Loading states from {}'.format(args.start_from_checkpoint))
            checkpoint = torch.load(args.start_from_checkpoint)
            fg_state = checkpoint['fg_best_state']
            fn_state = checkpoint['fn_best_state']
            film_generator.load_state_dict(fg_state)
            filmed_net.load_state_dict(fn_state)

        criterion = CrossEntropyLoss().cuda()

        fg_optimizer = Adam(film_generator.parameters(),
                            lr=args.lr,
                            weight_decay=1e-5)
        fn_optimizer = Adam(filmed_net.parameters(),
                            lr=args.lr,
                            weight_decay=1e-5)
        t = 0
        best_accuracy = 0
        for epoch in range(args.epochs):
            print ('Starting Epoch {}'.format(epoch))

            film_generator.cuda()
            filmed_net.cuda()
            film_generator.train()
            filmed_net.train()
            running_loss = 0

            for batch in train_loader:
                t += 1
                questions, feats, answers = batch
                questions = Variable(questions.cuda())
                feats = Variable(feats.cuda())
                answers = Variable(answers.cuda())

                fg_optimizer.zero_grad()
                fn_optimizer.zero_grad()

                film = film_generator(questions)
                output = filmed_net(feats, film)

                loss = criterion(output, answers)

                loss.backward()

                fg_optimizer.step()
                fn_optimizer.step()

                running_loss += loss.data[0]
                if t % 100 == 0:
                    print (t, running_loss / 100)

            film_generator.eval()
            filmed_net.eval()

            tr_accuracy = check_accuracy(film_generator,
                                         filmed_net,
                                         train_loader)
            print ('Epoch {}. Training accuracy:   {}'.
                    format(epoch, tr_accuracy))

            val_accuracy = check_accuracy(film_generator,
                                          filmed_net,
                                          val_loader)
            print ('Epoch {}. Validation accuracy: {}'.
                    format(epoch, val_accuracy))


            if val_accuracy >= best_accuracy:
                fg_best_state = get_state(film_generator)
                fn_best_state = get_state(filmed_net)


            checkpoint = {
            'fg_best_state' : fg_best_state,
            'fn_best_state' : fn_best_state,
            'epoch' : epoch,
            'val_accuracy' : val_accuracy,
            'vocab' : vocab
            }

            print ('Saving checkpoint to {}'.format(args.save_checkpoint_to))
            torch.save(checkpoint, args.save_checkpoint_to)


def check_accuracy(film_generator, filmed_net, loader):
    total, correct = 0, 0
    for batch in loader:
        questions, feats, answers = batch
        questions = Variable(questions.cuda(), volatile=True)
        feats = Variable(feats.cuda(), volatile=True)

        film = film_generator(questions)
        output = filmed_net(feats, film)

        _, pred = output.data.cpu().max(1)

        correct += (answers == pred).sum()
        total += questions.size(0)

    accuracy = correct/total
    return accuracy

def get_state(m):
    state={}
    for key, value in m.state_dict().items():
        state[key] = value.clone()
    return state

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
