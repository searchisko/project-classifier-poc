import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

import re
contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')


# cleaner (order matters)
def clean(text):
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

# sentence splitter
alteos = re.compile(r'([!\?])')


def sentences(l):
    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")

from zipfile import ZipFile
import json


def YelpReviews(label):
    with ZipFile("yelp_%s_set.zip"%label, 'r') as zf:
        with zf.open("yelp_%s_set/yelp_%s_set_review.json"%(label,label)) as f:
            for line in f:
                rev = json.loads(line)
                yield {'y':rev['stars'],\
                       'x':[clean(s).split() for s in sentences(rev['text'])]}

YelpReviews("test").next()

revtrain = list(YelpReviews("training"))
print len(revtrain), "training reviews"

## and shuffle just in case they are ordered
import numpy as np
np.random.shuffle(revtrain)


def StarSentences(reviews, stars=[1,2,3,4,5]):
    for r in reviews:
        if r['y'] in stars:
            for s in r['x']:
                yield s

from gensim.models import Word2Vec
import multiprocessing

## create a w2v learner
basemodel = Word2Vec(
    workers=multiprocessing.cpu_count(), # use your cores
    iter=3, # iter = sweeps of SGD through the data; more is better
    hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
    )
print basemodel

basemodel.build_vocab(StarSentences(revtrain))

from copy import deepcopy
starmodels = [deepcopy(basemodel) for i in range(5)]
for i in range(5):
    slist = list(StarSentences(revtrain, [i+1]))
    print i+1, "stars (", len(slist), ")"
    starmodels[i].train(  slist, total_examples=len(slist) )

"""
docprob takes two lists
* docs: a list of documents, each of which is a list of sentences
* models: the candidate word2vec models (each potential class)

it returns the array of class probabilities.  Everything is done in-memory.
"""

import pandas as pd # for quick summing within doc


def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
    # now exponentiate to get likelihoods,
    lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame((lhd/lhd.sum(axis=0)).transpose())
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i, d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob

# read in the test set
revtest = list(YelpReviews("test"))

# get the probs (note we give docprob a list of lists of words, plus the models)
probs = docprob( [r['x'] for r in revtest], starmodels )

probpos = pd.DataFrame({"out-of-sample prob positive": probs[[3, 4]].sum(axis=1),
                        "true stars": [r['y'] for r in revtest]})
probpos.boxplot("out-of-sample prob positive", by="true stars", figsize=(12, 5))
