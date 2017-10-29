import numpy as np
import ipdb as pdb
import h5py
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class MyDataset(Dataset):
    def __init__(self, feats_h5, questions_h5):
        self.feats_h5 = feats_h5
        self.questions = self._dset_to_tensor(questions_h5['questions'])
        self.image_idxs = self._dset_to_tensor(questions_h5['image_idxs'])

        #If there are answers, take them
        self.answers = None
        if 'answers' in questions_h5:
            self.answers = self._dset_to_tensor(questions_h5['answers'])

    def _dset_to_tensor(self, dset):
        tensor = LongTensor(np.asarray(dset, np.int64))
        return tensor

    def __getitem__(self, idx):
        """ Get one item"""
        question = self.questions[idx]
        image_idx = self.image_idxs[idx]
        answer = None
        if isinstance(self.answers, LongTensor):
            answer = self.answers[idx]

        feats = self.feats_h5['features'][image_idx]
        feats = FloatTensor(np.asarray(feats, np.float32))

        return question, feats, answer

    def __len__(self):
        return self.questions.size(0)


class MyDataLoader(DataLoader):
    def __init__(self, feats_h5_path, questions_h5_path):
        self.feats_h5 = h5py.File(feats_h5_path, 'r')
        with h5py.File(questions_h5_path, 'r') as questions_h5:
            self.dataset = MyDataset(self.feats_h5, questions_h5)

        super().__init__(self.dataset, batch_size=64, shuffle=True,
                                    collate_fn=my_collate)

    #Making sure that h5 file is closed correctly
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.feats_h5:
            self.feats_h5.close()

def my_collate(batch):
    """Custome collate function to deal with None in answers"""
    #If data field is not None collate it.
    stacked = list(zip(*batch))

    question_batch = default_collate(stacked[0])

    feats_batch = default_collate(stacked[1])

    answer_batch = stacked[2]
    if stacked[2][0] is not None:
        answer_batch = default_collate(stacked[2])
    return question_batch, feats_batch, answer_batch
