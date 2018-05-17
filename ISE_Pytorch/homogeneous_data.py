import numpy
import copy
import IPython


class HomogeneousData():

    def __init__(self, data, batch_size=128, maxlen=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]
        self.feats = self.data[1]  # todo will be images and have to put model to get features here

        assert len(self.caps) == len(self.data[2])

        # find the unique lengths
        self.lengths = [len(cc.split()) for cc in self.caps]
        # determine the word count of each sentence
        self.len_unique = numpy.unique(self.lengths)  # find unique numbers of word counts in sentence

        # indices of unique sen_lengths
        self.len_indices = dict()  # dictionary of unique lengths(x) as keys, and the indices of sentences of len x
        self.len_counts = dict()  # dict of unique lengths and thier count
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]  # all indices of sentences of length x
            self.len_counts[ll] = len(self.len_indices[ll])  # number of sentences with a aprticular word count

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[
                ll] = 0  # get some unique index of words of length x-- pointers to how many items taken from every sentence count
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def __next__(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx + 1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch of sentences of similar word count
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos + curr_batch_size]
        self.len_indices_pos[self.len_unique[
            self.len_idx]] += curr_batch_size  # pointers to where we arein the specific indices of a certain sentence count
        self.len_curr_counts[self.len_unique[
            self.len_idx]] -= curr_batch_size  # update how many sentences are left for an index of a certain sentence count

        caps = [self.caps[ii] for ii in curr_indices]
        genre = [self.data[3][ii] for ii in curr_indices]
        artist = [self.data[2][ii] for ii in curr_indices]
        feats = [self.feats[ii] for ii in curr_indices]

        return caps, feats, artist, genre  # return a minbactch <= 128 of sentences with euqual lengths. and image features

    def __iter__(self):
        return self


def prepare_data(caps, img_features, artist, genre, worddict, maxlen=None, n_words=10000):
    """
    Put data into format useable by the model (ordered by count of words in sent, nsamples)
    """
    seqs, seqs1, seqs2 = [], [], []
    feat_list = []
    for i, (cc, art, gen) in enumerate(zip(caps, artist, genre)):  # feed genre and styles as list
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in
                     cc.split()])  # list of list of indices of words for sentences
        seqs1.append([worddict[a] if worddict[a] < n_words else 1 for a in art.split()])
        seqs2.append([worddict[g] if worddict[g] < n_words else 1 for g in gen.split()])
        feat_list.append(img_features[i])

    all_lengths = [(len(s), len(s1), len(s2)) for (s, s1, s2) in zip(seqs, seqs1, seqs2)]

    lengths = list(map((lambda t: t[0]), all_lengths))
    lena = list(map((lambda t: t[1]), all_lengths))
    leng = list(map((lambda t: t[2]), all_lengths))

    y = numpy.asarray(feat_list, dtype=numpy.float32)

    n_samples = len(seqs)
    maxlen = numpy.max(lengths) + 1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],
        idx] = s  # put the list of word indices of sentence in array(x) max_number-of-words, samples, 0 paddes

    aa = numpy.zeros((numpy.max(lena) + 1, len(seqs1))).astype('int64')
    for idx, s in enumerate(seqs1):
        aa[:lena[idx], idx] = s

    gg = numpy.zeros((numpy.max(leng) + 1, len(seqs2))).astype('int64')
    for idx, s in enumerate(seqs2):
        gg[:leng[idx], idx] = s

    return x, y, aa, gg  # (max_len_sen, nsamples) , feature_list, (max_len_artist,n) (max_len_genre,n)
