# -*- coding: utf-8 -*-

import torch
import logging
import os
import pickle as pkl

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import load_dataset
from vocab import build_dictionary
import homogeneous_data
from torch.autograd import Variable
import time
from model import Img_Sen_Artist_Ranking, PairwiseRankingLoss
import numpy
from tools_ import encode_sentences, encode_images
from evaluation import i2t


logging.basicConfig(level=logging.INFO)

def trainer(data='f30k',
            margin=0.2,
            dim=1024,
            dim_image=4096,
            dim_word=300,
            max_epochs=15,
            encoder='lstm',
            dispFreq=10,
            grad_clip=2.0,
            maxlen_w=150,
            batch_size=128,
            saveto='vse/f30K',
            validFreq=100,
            early_stop=20,
            lrate=1e-3,
            reload_=False):
    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_

    logging.info(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        logging.info('reloading...' + saveto)
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    # Load training and development sets
    logging.info('loading dataset')
    titles, album_ims, artist, genre = load_dataset(data)
    artist_string = artist
    genre_string = genre

    # Create and save dictionary
    if os.path.exists('%s.dictionary.pkl' % saveto):
        logging.info('loading dict from...' + saveto)
        with open('%s.dictionary.pkl' % saveto, 'rb') as wdict:
            worddict = pkl.load(wdict)
        n_words = len(worddict)
        model_options['n_words'] = n_words
        logging.info('Dictionary size: ' + str(n_words))
    else:

        logging.info('Create dictionary')
        worddict = build_dictionary(titles + artist + genre)[0]
        n_words = len(worddict)
        model_options['n_words'] = n_words
        logging.info('Dictionary words: ' + str(n_words))
        with open('%s.dictionary.pkl' % saveto, 'wb') as f:
            pkl.dump(worddict, f)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.items():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    model_options['worddict'] = worddict
    model_options['word_idict'] = word_idict

    # Each sentence in the minibatch have same length (for encoder)
    train_iter = homogeneous_data.HomogeneousData([titles, album_ims, artist, genre], batch_size=batch_size,
                                                  maxlen=maxlen_w)

    img_sen_model = Img_Sen_Artist_Ranking(model_options)
    # todo code to load saved model dict
    if os.path.exists('%s_model_%s.pkl' % (saveto, encoder)):
        logging.info('Loading model...')
        # pkl.dump(model_options, open('%s_params_%s.pkl' % (saveto, encoder), 'wb'))
        img_sen_model.load_state_dict(torch.load('%s_model_%s.pkl' % (saveto, encoder)))
        logging.info('Done')
    img_sen_model = img_sen_model.cuda()

    loss_fn = PairwiseRankingLoss(margin=margin).cuda()

    params = filter(lambda p: p.requires_grad, img_sen_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lrate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=40, mode='min',
                                  verbose=True, threshold=1e-8)

    uidx = 0
    curr = 0.0
    n_samples = 0

    # For Early-stopping
    best_r1, best_r5, best_r10, best_medr = 0.0, 0.0, 0.0, 0
    best_step = 0

    writer = SummaryWriter()
    for eidx in range(max_epochs):

        for x, im, artist, genre in train_iter:
            n_samples += len(x)
            uidx += 1

            x, im, artist, genre = homogeneous_data.prepare_data(x, im, artist, genre, worddict, maxlen=maxlen_w,
                                                                 n_words=n_words)

            if x is None:
                logging.info('Minibatch with zero sample under length ', maxlen_w)
                uidx -= 1
                continue

            x = Variable(torch.from_numpy(x).cuda())
            im = Variable(torch.from_numpy(im).cuda())
            artist = Variable(torch.from_numpy(artist).cuda())
            genre = Variable(torch.from_numpy(genre).cuda())
            # Update
            x1, im1, artist, genre = img_sen_model(x, im, artist, genre)

            #make validation on inout before trainer see it
            if numpy.mod(uidx, validFreq) == 0:
                with torch.no_grad():
                    print('Epoch ', eidx, '\tUpdate@ ', uidx, '\tCost ', cost.data.item())
                    writer.add_scalar('Evaluation/Validation_Loss', cost.data.item(), uidx)
                    (r1, r5, r10, medr) = i2t(im1, x) #distances with l2norm
                    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))

                    (r1g, r5g, r10g, medrg) = i2t(im1, genre)
                    logging.info("Image to genre: %.1f, %.1f, %.1f, %.1f" % (r1g, r5g, r10g, medrg))

                    (r1a, r5a, r10a, medra) = i2t(im1, artist)
                    logging.info("Image to Artist: %.1f, %.1f, %.1f, %.1f" % (r1a, r5a, r10a, medra))

                    logging.info("Cal Recall@K ")
                    writer.add_scalars('Validation Recal/Image2Album', {'r@1':r1 ,
                                                                         'r@5': r5,
                                                                         'r@10': r10}, uidx)

                    writer.add_scalars('Validation Recal/Image2Genres', {'r@1': r1g,
                                                                         'r@5': r5g,
                                                                         'r@10': r10g}, uidx)

                    writer.add_scalars('Validation Recal/Image2Artist', {'r@1': r1a,
                                                                           'r@5': r5a,
                                                                           'r@10': r5a}, uidx)

                    curr_step = uidx / validFreq

                    currscore = r1 + r5 + r10 + r1a + r5a + r10a + r1g + r5g + r10g-medr-medrg-medra
                    if currscore > curr:
                        curr = currscore
                        best_r1, best_r5, best_r10, best_medr = r1, r5, r10, medr
                        best_r1g, best_r5g, best_r10g, best_medrg = r1, r5, r10, medrg
                        best_step = curr_step

                        # Save model
                        logging.info('Saving model...')
                        pkl.dump(model_options, open('%s_params_%s.pkl' % (saveto, encoder), 'wb'))
                        torch.save(img_sen_model.state_dict(), '%s_model_%s.pkl' % (saveto, encoder))
                        logging.info('Done')

                    if curr_step - best_step > early_stop:
                        logging.info('early stopping, jumping now...')
                        logging.info("Image to text: %.1f, %.1f, %.1f, %.1f" % (best_r1, best_r5, best_r10, best_medr))
                        logging.info("Image to genre: %.1f, %.1f, %.1f, %.1f" % (best_r1g, best_r5g, best_r10g, best_medrg))

                        #return 0
                        lrate = 1e-4
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lrate
            cost = loss_fn(im1, x, artist, genre)
            writer.add_scalar('Evaluation/training_Loss', cost, uidx)

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

            scheduler.step(cost.data.item())
            optimizer.step()

        #scheduler.step(cost.data.item())
        logging.info('Seen %d samples' % n_samples)





if __name__ == '__main__':
    pass
