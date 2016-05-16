import os
import scipy.io.wavfile as wav
import numpy as np
import fnmatch
import pickle
import tables
import csv
import theano
from audio_tools import wdct, iwdct

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

def fetch_blizzard(sz=100):
    partial_path = get_dataset_dir("blizzard")
    hdf5_path = os.path.join(partial_path, "blizzard_{}_saved.hdf5".format(sz))

    all_transcripts = []
    pcms = []
    pcms_lens = []
    mx = np.zeros(64)
    all_chars = ([chr(ord('a') + i) for i in range(26)] +
                 [',', '.', '!', '?', '<UNK>'])
    
    code2char = dict(enumerate(all_chars))
    char2code = {v: k for k, v in code2char.items()}
    vocabulary_size = len(char2code.keys())
    fft_size = 64

    if not os.path.exists(hdf5_path):
        print ('Building', hdf5_path, 'fft size', fft_size, 'vocab size', vocabulary_size)
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data_storage = hdf5_file.create_vlarray(hdf5_file.root, 'data', tables.Float32Atom(shape=(fft_size)),
                                                "audio representation ",
                                                filters=tables.Filters(1))
        length_storage = hdf5_file.create_vlarray(hdf5_file.root, 'length',
                                                  tables.Int32Atom(shape=()),
                                                  "wav len in samples",
                                                  filters=tables.Filters(1))
        offset_storage = hdf5_file.create_vlarray(hdf5_file.root, 'offset',
                                                  tables.Int64Atom(shape=()),
                                                  "wav len in samples",
                                                  filters=tables.Filters(1))
        text_storage = hdf5_file.create_vlarray(hdf5_file.root, 'text',
                                                tables.StringAtom(itemsize=2),
                                                "ragged array of strings",
                                                filters=tables.Filters(1))
        text_storage.flavor = 'python'
        
        text_1hot_storage = hdf5_file.create_vlarray(hdf5_file.root, 'text_1hot', tables.Float32Atom(shape=(vocabulary_size)),
                                                "text one hot encoded ",
                                                filters=tables.Filters(1))

        rng = np.random.RandomState(1999)
        with open(os.path.join(partial_path, "prompts_{}.txt".format(sz)), 'rb') as f_p:
            utts = csv.reader(f_p, delimiter='\t')
            l=list(utts)
            rng.shuffle(l)
            for u in l:
                print ("{}.wav".format(u[0]), u[1])
                fs, w = wav.read(os.path.join(partial_path, "wavn_{}".format(sz), "{}.wav".format(u[0])))
                print w.shape
                if fs != 8000 and fs != 16000:
                    print "Unexpected sampling rate in %s" % u[0]
                    raise
                if fs == 16000:
                    w = w[::2]
                w_i = wdct(w)
                data_storage.append(w_i)
                text_storage.append(u[1].lower())
                text_1hot_storage.append(tokenize_ind(u[1].lower(), char2code))
        hdf5_file.close()
        
    hdf5_file = tables.openFile(hdf5_path, mode='r')


        #all_pcm_max = [np.amax(np.abs(b), axis=0) for b in pcms]
        #mx = np.max(np.vstack(all_pcm_max), axis = 0) 
        #pcms = [ p / mx for p in pcms ]

        #num_examples = len(pcms)
        # Shuffle for train/validation/test division
        #rng = np.random.RandomState(1999)
        #shuffle_idx = rng.permutation(num_examples)

        #pcms = [pcms[x] for x in shuffle_idx]
        #all_transcripts = [all_transcripts[x] for x in shuffle_idx]



    pickle_dict = {}
    pickle_dict["data"] = hdf5_file.root.data
    pickle_dict["target_phrases"] = hdf5_file.root.text
    pickle_dict["vocabulary_size"] = vocabulary_size
    pickle_dict["vocabulary_tokenizer"] = tokenize_ind
    pickle_dict["vocabulary"] = char2code
    pickle_dict["target"] = hdf5_file.root.text_1hot
    return pickle_dict
        

class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 make_mask=False,
                 one_hot_class_size=None,
                 shuffle=True):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.slice_start_ = start_index
        self.axis = axis
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
            num_examples = len(self.list_of_containers[0]) - self.start_index
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
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        if self.shuffle :
            ind = self.shuffle_idx[self.slice_start_:self.slice_end_]
        else:
            ind = slice(self.slice_start_, self.slice_end_)
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

        
if __name__ == "__main__":
    import argparse
    import theano
    d = fetch_blizzard(sz='fruit')
    X = d["data"]
    y = d["target"]
    vocabulary = d["vocabulary"]
    vocabulary_size = d["vocabulary_size"]
    minibatch_size = 2
    train_itr = list_iterator([X, y], minibatch_size, axis=1, stop_index=20,
                              make_mask=True, shuffle=True)
    valid_itr = list_iterator([X, y], 20, axis=1, start_index=80,
                              make_mask=True, shuffle=False)
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr)
    except StopIteration:
        pass
    print "done first pass"
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(train_itr)
    except StopIteration:
        pass
    print "done second pass"
    try:
        while True:
            X_mb, X_mb_mask, c_mb, c_mb_mask = next(valid_itr)
    except StopIteration:
        pass
    print "done valid pass"

    from IPython import embed; embed()
    raise ValueError()
