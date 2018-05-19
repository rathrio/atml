import numpy
import torch
from ISE_Pytorch.datasets import load_dataset
from ISE_Pytorch.tools_ import encode_sentences, embbed_sentences, encode_images


def evalrank(model ):
    """
    Evaluate a trained model on either dev or test
    """

    print('Loading dataset')

    titles, album_ims, _, _ = load_dataset()

    print('Computing results...')
    ls = encode_sentences(model, titles)
    lim = encode_images(model, album_ims)

    (r1, r5, r10, medr) = i2t(lim, ls)
    print("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
    (r1i, r5i, r10i, medri) = t2i(lim, ls)
    print("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))

def get_embedding(model, input):
    """
    Evaluate a trained model on  some input
    get input , chop them and then send to lstm
    """
    art_emb = embbed_sentences(model, input)

    return art_emb



def i2t(images, captions, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] // 5

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())  # multiply a set of image norms and their caption norms
        _, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()  # list of desending sorted high scores

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i(images, captions, npts=None, data='f8k'):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] // 5

    ims = torch.cat([images[i].unsqueeze(0) for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index: 5 * index + 5]

        # Compute scores
        d = torch.mm(queries, ims.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[5 * index + i] = numpy.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


