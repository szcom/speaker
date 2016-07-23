import os
from numpy.lib.stride_tricks import as_strided
import scipy.io.wavfile as wav
import numpy as np
import fnmatch
import pickle
import tables
import csv
import theano
import time
from audio_tools import wdct, iwdct
from kdllib import apply_quantize_preproc

def get_dataset_dir(dataset_name):
    """ Get dataset directory path """
    return os.sep.join(os.path.realpath(__file__).split
                       (os.sep)[:-1] + [dataset_name])
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dense = labels_dense.reshape([-1])
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(num_classes,))
    return labels_one_hot

def tokenize_ind(phrase, vocabulary):
    vocabulary_size = len(vocabulary.keys())
    filter_ = [char_ in vocabulary.keys() for char_ in phrase]
    phrase = [vocabulary[char_] for char_, cond in zip(phrase, filter_) if cond]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = dense_to_one_hot(phrase, vocabulary_size)
    return phrase

def fetch_blizzard(sz=100, frame_size=4):
    partial_path = get_dataset_dir("blizzard")
    hdf5_path = os.path.join(partial_path, "blizzard_{}_saved.hdf5".format(sz))

    all_transcripts = []
    all_chars = ([chr(ord('a') + i) for i in range(26)] +
                 [',', '.', '!', '?', '<UNK>'])
    
    code2char = dict(enumerate(all_chars))
    char2code = {v: k for k, v in code2char.items()}
    vocabulary_size = len(char2code.keys())

    if not os.path.exists(hdf5_path):
        print ('Building', hdf5_path, 'frame size', frame_size, 'vocab size', vocabulary_size)
        filters = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        chunk_depth = 64*1000
        chunked_data_storage = hdf5_file.createEArray(hdf5_file.root, 'chunked_data',
                                                      tables.Float32Atom(),
                                                      shape=(0, chunk_depth),
                                                      filters=filters)

        data_storage = hdf5_file.create_vlarray(hdf5_file.root, 'data', tables.Float32Atom(shape=(frame_size)),
                                                "audio representation ",
                                                filters=filters)
        length_storage = hdf5_file.create_vlarray(hdf5_file.root, 'length',
                                                  tables.Int32Atom(shape=()),
                                                  "wav len in samples")
        offset_storage = hdf5_file.create_vlarray(hdf5_file.root, 'offset',
                                                  tables.Int64Atom(shape=()),
                                                  "wav len in samples")
        text_storage = hdf5_file.create_vlarray(hdf5_file.root, 'text',
                                                tables.StringAtom(itemsize=2),
                                                "ragged array of strings")
        text_storage.flavor = 'python'
        
        text_1hot_storage = hdf5_file.create_vlarray(hdf5_file.root, 'text_1hot', tables.Float32Atom(shape=(vocabulary_size)),
                                                "text one hot encoded ",
                                                filters=tables.Filters(1))

        rng = np.random.RandomState(1999)
        seq_offset = 0
        tail = np.zeros((1, chunk_depth))
        with open(os.path.join(partial_path, "prompts_{}.txt".format(sz)), 'rb') as f_p:
            utts = csv.reader(f_p, delimiter='\t')
            l=list(utts)
            l=sorted(l, key=lambda u: len(u[1]))
            for n, u in enumerate(l):
                print ("{}.wav".format(u[0]), u[1], len(u[1]))
                fs, w = wav.read(os.path.join(partial_path, "wavn_{}".format(sz), "{}.wav".format(u[0])))
                print (w.shape)
                if fs != 8000 and fs != 16000:
                    print ("Unexpected sampling rate in %s" % u[0])
                    raise
                if fs == 16000:
                    w = w[::2]
                #remove DC
                w = w - w.mean()    
                append = np.zeros((frame_size - len(w) % frame_size))
                w = np.hstack((w,append))
                w_i = apply_quantize_preproc([w])[0].reshape((-1, frame_size))
                # not chunked auido is only for tests
                if n < 1000:
                    data_storage.append(w_i)
                length_storage.append([len(w_i)])
                text_storage.append(u[1].lower())
                text_1hot_storage.append(tokenize_ind(u[1].lower(), char2code))
                offset_storage.append([seq_offset])
                z = as_strided(w_i, strides=(w_i.itemsize*chunk_depth, w_i.itemsize), shape = [np.prod(w_i.shape)//chunk_depth, chunk_depth] )
                chunked_data_storage.append(z)
                seq_offset += len(z)
                r = np.prod(w_i.shape) - np.prod(z.shape)
                if r > 0:
                    tail[0, :r] = w_i.reshape(-1)[-r:]
                    tail[0, r:] = 1
                    seq_offset += 1
                    chunked_data_storage.append(tail)
        offset_storage.append([seq_offset])
        hdf5_file.close()
        
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    pickle_dict = {}
    pickle_dict["data"] = hdf5_file.root.data
    pickle_dict["data_offset"] = hdf5_file.root.offset
    pickle_dict["chunked_data"] = hdf5_file.root.chunked_data
    pickle_dict["data_length"] = hdf5_file.root.length
    pickle_dict["target_phrases"] = hdf5_file.root.text
    pickle_dict["vocabulary_size"] = vocabulary_size
    pickle_dict["vocabulary_tokenizer"] = tokenize_ind
    pickle_dict["vocabulary"] = char2code
    pickle_dict["target"] = hdf5_file.root.text_1hot
    return pickle_dict
        

class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index = 0,
                 stop_index = np.inf,
                 make_mask = False,
                 one_hot_class_size = None,
                 shuffle=True,
                 frame_size = 4):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.slice_start_ = start_index
        self.axis = axis
        self.frame_size = frame_size
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)
        self.one_hot_class_size = one_hot_class_size
        if one_hot_class_size is not None:
            assert len(self.one_hot_class_size) == len(list_of_containers)
            
        self.rng = np.random.RandomState(1999)
        self.shuffle=shuffle
        self.reset()
    def reset(self):
        if self.stop_index == np.inf:
            num_examples = min([len(i) for i in self.list_of_containers]) - self.start_index
            self.stop_index = num_examples + self.start_index
        else:
            num_examples = self.stop_index - self.start_index
        self.shuffle_idx = np.append(np.zeros(self.start_index, dtype=np.int), self.rng.permutation(num_examples) + self.start_index)
        self.slice_start_ = self.start_index
    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        #print(self.slice_end_, self.stop_index)
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        if self.shuffle :
            ind = self.shuffle_idx[self.slice_start_:self.slice_end_]
        else:
            ind = np.arange(self.slice_start_, self.slice_end_)
        #print ('ind', ind, self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            res = self._slice_without_masks(ind)
            if not all([self.minibatch_size in r.shape for r in res]):
                # TODO: Check that things are even
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res
        else:
            res = self._slice_with_masks(ind)
            # TODO: Check that things are even
            if not all([self.minibatch_size in r.shape for r in res]):
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res

    def _slice_without_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")

    def _slice_with_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        sliced_c = [c[ind] for c in self.list_of_containers]
        if min([len(i) for i in sliced_c]) < self.minibatch_size:
            self.reset()
            raise StopIteration("Invalid length slice")
        for n in range(len(sliced_c)):
            sc = sliced_c[n]
            if self.one_hot_class_size is not None:
                convert_it = self.one_hot_class_size[n]
                if convert_it is not None:
                    raise ValueError("One hot conversion not implemented")
            if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                
                maxlen = max([len(i) for i in sc])
                # Assume they at least have the same internal dtype
                if len(sc[0].shape) > 1:
                    total_shape = (maxlen, sc[0].shape[1])
                elif len(sc[0].shape) == 1:
                    total_shape = (maxlen, 1)
                else:
                    raise ValueError("Unhandled array size in list")
                if self.axis == 0:
                    raise ValueError("Unsupported axis of iteration")
                    new_sc = np.zeros((len(sc), total_shape[0],
                                       total_shape[1]))
                    new_sc = new_sc.squeeze().astype(sc[0].dtype)
                else:
                    new_sc = np.zeros((total_shape[0], len(sc),
                                       total_shape[1]))
                    new_sc = new_sc.astype(sc[0].dtype)
                    for m, sc_i in enumerate(sc):
                        new_sc[:len(sc_i), m, :] = sc_i
                sliced_c[n] = new_sc
        return sliced_c

    def _mask_like(self, alike, sc):
        rv = np.ones_like(alike)
        for m, sc_i in enumerate(sc):
            rv[len(sc_i):, m] = 0
        return rv
    def _slice_with_masks(self, ind):
        #cs - list of arrays
        cs = self._slice_without_masks(ind)
        sliced_c = [c[ind] for c in self.list_of_containers]
        if self.axis == 0:
            ms = [self._mask_like(c[:, 0], corg) for c, corg in zip(cs, sliced_c)]
        elif self.axis == 1:
            ms = [self._mask_like(c[:, :, 0], corg) for c, corg in zip(cs, sliced_c)]
        assert len(cs) == len(ms)
        return [i for sublist in list(zip(cs, ms)) for i in sublist]
    
class pair_audio_oh_iterator(base_iterator):
    def set_audio_length_and_offset(self, length, offset):
        self.length = length
        self.offset = offset
    def _slice_without_masks(self, ind):
        assert False
    def _chunk_indexes(self, seq_ind):
        rv = {}
        for i in seq_ind:
            rv[i] = range(self.offset[i], self.offset[i+1])
        return rv
    def _slice_with_masks(self, ind):
        #print ([c.shape for c in self.list_of_containers])
        oh_txt = self.list_of_containers[1]

        chunk_ind = self._chunk_indexes(ind)
        
        audio_maxlen = max(self.length[ind])
        #print ('audio mlen:', audio_maxlen)
        oh_maxlen = max([len(oh_txt[i]) for i in ind])
        #print ('oh mlen:', oh_maxlen, oh_txt.dtype)
        audio = self.list_of_containers[0]
        audio_sc = np.zeros((audio_maxlen, len(ind), self.frame_size), dtype=audio[0].dtype)
        audio_sc_mask = np.zeros((audio_maxlen, len(ind)), dtype=audio[0].dtype)
        oh_depth = oh_txt[0].shape[1]
        oh_sc = np.zeros((oh_maxlen, len(ind), oh_depth),  dtype=oh_txt[0].dtype)
        oh_sc_mask = np.zeros((oh_maxlen, len(ind)), dtype=oh_txt[0].dtype)
        for b, i in enumerate(ind):
            p = 0
            for j in chunk_ind[i]:
                audio_j = audio[j].reshape(-1, self.frame_size)
                q = p + audio_j.shape[0]
                q = min(q, self.length[i])
                audio_sc[p:q, b ] = audio_j[:q-p]
                audio_sc_mask[p:q, b ] = 1.
                p += audio_j.shape[0]
            oh_sc[:len(oh_txt[i]),b] = oh_txt[i]
            oh_sc_mask[:len(oh_txt[i]), b ] = 1.
        return audio_sc, audio_sc_mask, oh_sc, oh_sc_mask

def test_iter(d):
    X = d["data"]
    X_c = d["chunked_data"]
    X_l = d["data_length"]
    y = d["target"]
    vocabulary = d["vocabulary"]
    vocabulary_size = d["vocabulary_size"]
    minibatch_size = 21
    train_itr = list_iterator([X, y], minibatch_size, axis=1, stop_index=800,
                              make_mask=True, shuffle=False)
    train_itr2 = pair_audio_oh_iterator([X_c, y], minibatch_size, axis=1, stop_index=800,
                                       make_mask=True, shuffle=False)
    train_itr2.set_audio_length_and_offset(d["data_length"], d["data_offset"])
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr2)
            X_mb1, X_mb_mask1, c_mb1, c_mb_mask1 = next(train_itr)
            print(np.where(X_mb_mask != X_mb_mask1) )
            print(X_mb_mask.shape, X_mb_mask1.shape)
            #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
            #raise ValueError()
            assert np.array_equal(X_mb, X_mb1) 
            assert np.array_equal(X_mb_mask, X_mb_mask1) 
            assert np.array_equal(c_mb, c_mb1) 
            assert np.array_equal(c_mb_mask, c_mb_mask1) 
    except StopIteration:
        pass
    
    
if __name__ == "__main__":
    import argparse
    import theano
    d = fetch_blizzard(sz='fruit')
    test_iter(d)
    X = d["data"]
    X_c = d["chunked_data"]
    X_l = d["data_length"]
    y = d["target"]
    vocabulary = d["vocabulary"]
    vocabulary_size = d["vocabulary_size"]
    minibatch_size = 50


    train_itr = list_iterator([X, y], minibatch_size, axis=1, stop_index=800,
                              make_mask=True, shuffle=True)
    train_itr2 = pair_audio_oh_iterator([X_c, y], minibatch_size, axis=1, stop_index=800,
                              make_mask=True, shuffle=True)
    train_itr2.set_audio_length_and_offset(d["data_length"], d["data_offset"])
    valid_itr = list_iterator([X, y], minibatch_size, axis=1, start_index=800,
                              make_mask=True, shuffle=False)
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(valid_itr)
    except StopIteration:
        pass
    print ("done valid pass")
    start = time.time()
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr2)
    except StopIteration:
        pass
    
    print ("done first pass, took:")
    print (time.time() - start)
    start = time.time()
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr)
    except StopIteration:
        pass
    print ("done second pass, took :")
    print (time.time() - start)

    from IPython import embed; embed()
    raise ValueError()
