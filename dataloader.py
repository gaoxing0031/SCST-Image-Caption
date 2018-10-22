import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data
from functools import reduce

from opts import Opt

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.opt.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.opt.seq_length

    def __init__(self, opt):
        self.opt = opt

        # Load the json file which contains additional information about dataset
        print('DataLoader loading json file : {}'.format(self.opt.input_json_file))
        self.info = json.load(open(self.opt.input_json_file))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        opt.vocab_size = self.vocab_size
        assert opt.vocab_size > 0
        print('Vocab size is {}'.format(opt.vocab_size))

        # Load h5py file
        print('DataLoader loading h5py file : {}'.format(self.opt.input_label_file))
        self.h5_label_file = h5py.File(self.opt.input_label_file, 'r', driver = 'core')
        seq_size = self.h5_label_file['labels'].shape # [6W, 16]
        opt.seq_length = seq_size[1]
        assert opt.seq_length > 0
        print('Length of seq is {}'.format(opt.seq_length))
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        assert self.num_images > 0
        print('Number of images is {}'.format(self.num_images))

        self.split_ix = {'train':[], 'val':[], 'test':[]}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            else:
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)
    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.opt.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.opt.seq_length]
        else: #ncap == seq_per_img
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.opt.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.opt.batch_size
        seq_per_img = seq_per_img or self.opt.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.opt.seq_length + 2], dtype = 'int') # Add <START> <END>
        mask_batch = np.zeros([batch_size * seq_per_img, self.opt.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            tmp_fc, tmp_att, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            label_batch[i*seq_per_img:(i+1)*seq_per_img, 1:self.opt.seq_length+1] = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        fc_batch, att_batch, label_batch, gts, infos = zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x:0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_] * seq_per_img for _ in fc_batch]))

        max_att_len = max([_.shape[0] for _ in att_batch])
        # data['att_feats'] = np.zeros([len(att_batch)*seq_per_img, max_att_len, att_batch[0].shape[1]], dtype='float32')
        data['att_feats'] = np.zeros([len(att_batch)*seq_per_img, att_batch[0].size(1), att_batch[0].size(2)])
        for i in range(len(att_batch)):
            # data['att_feats'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[1]] = att_batch[i]
            data['att_feats'][i*seq_per_img : (i+1)*seq_per_img, : , : ] = att_batch[i]
        # TODO: Figure out the principle to arrange att feats
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

        # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
        # so that the torch.utils.data.DataLoader can load the data according the index.
        # However, it's minimum change to switch to pytorch data loading.

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        opt = self.opt
        ix = index  # self.split_ix[index]
        if opt.use_att:
            att_feat = np.load(os.path.join(opt.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if opt.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
        else:
            att_feat = np.zeros((1, 1, 1))
        return (np.load(os.path.join(opt.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy')),
                att_feat,
                ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    # return iter([self.indices[i] for i in torch.randperm(self.num_samples).long()])

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
        return tmp + [wrapped]

if __name__ == '__main__':
    opt = Opt()
    loader = DataLoader(opt)
    data = loader.get_batch('train')
    import pickle
    with open('G:/data_self.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('SUCCESS!')