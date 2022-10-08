# coding=latin-1
import prosit
import argparse, os, re, sys, time, json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
import torch
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer
#import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['JOBLIB_MULTIPROCESSING'] = '0' # o runna in parallelo questo script diverse volte... ma perché???

###################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-device",              type=str,   default='cuda:1')
parser.add_argument("-path",                type=str,   default='data/Tweet.txt', help="reuters_trn.txt Tweet.txt")
parser.add_argument("-dir_data",            type=str,   default='data/')
parser.add_argument("-limit",               type=int,   default=50000, help="twitter: 2472, reuters: 7769, newsgroups: 11314, dbpedia: 20000")
parser.add_argument("-noabove",             type=float, default=.6)
parser.add_argument("-nobelow",             type=float, default=.001)
parser.add_argument("-alpha",               type=float, default=[0.0001],    nargs='+', help="")
parser.add_argument("-closest_texts",       type=float, default=[0.1], nargs='+', help="")
parser.add_argument("-mintopics",           type=int,   default=4)
parser.add_argument("-maxtopics",           type=int,   default=100)
parser.add_argument("-nr_descriptors",      type=int,   default=10)
parser.add_argument("-cv",                  type=bool,  default=True)
parser.add_argument("-npmi",                type=bool,  default=True)
parser.add_argument("-weco",                type=bool,  default=False)
parser.add_argument("-rbo",                 type=bool,  default=False)
parser.add_argument("-distinct",            type=bool,  default=True)
parser.add_argument("-verbose",             type=bool,  default=False)
parser.add_argument("-png",                 type=bool,  default=False)
parser.add_argument("-pngdims",             type=int,   default=3)
parser.add_argument("-pngpoints",           type=int,   default=1000)
parser.add_argument("-t2v_train",           type=str,   default='deep-learn', help="learn fast-learn deep-learn")
parser.add_argument("-epochs",              type=int,   default=100)
parser.add_argument("-epss",                type=float, default=[.01, .1, .3], nargs='+', help="[.0001, .00001]")
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
###################################################################################################


def file2list(pathfile, encoding='utf-8', elemtype=int, sep="\n", emptyend=True):
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    if emptyend: str_file = re.sub("\n+$", '', str_file) # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    if elemtype == int:
        out = [float(x) for x in str_file.split(sep)] # se il format è float, non lo traduce direttamente
        out = [int(x) for x in out]
    elif elemtype == float:
        out = [float(x) for x in str_file.split(sep)]
    else:
        out = [x for x in str_file.split(sep)]
    return out


def compute_prosit(txt, emb, method, dirout):
    tm = prosit.PST(alpha=args.alpha, max_topics=args.maxtopics, min_topics=args.mintopics, nr_descriptors=args.nr_descriptors, closest_texts_rate=args.closest_texts, dir_out=dirout, device=device, verbose=args.verbose)
    models = tm.fit(txt, emb, cv=args.cv, npmi=args.npmi, weco=args.weco, rbo=args.rbo, distinct=args.distinct, png=args.png, pngdims=args.pngdims, pngpoints=args.pngpoints)
    prosit_out = list()
    for alpha in models:
        for n_topics in models[alpha]:
            for texts_rate in models[alpha][n_topics]:
                prosit_out.append([f"prosit_{method}", alpha, texts_rate, n_topics, models[alpha][n_topics][texts_rate]['cv'], models[alpha][n_topics][texts_rate]['npmi'], models[alpha][n_topics][texts_rate]['weco'], models[alpha][n_topics][texts_rate]['rbo'], models[alpha][n_topics][texts_rate]['specif'], models[alpha][n_topics][texts_rate]['dissim'],
                '---'.join([' '.join(topic) for topic in models[alpha][n_topics][texts_rate]['descriptors']])])
                # for k in models[alpha][n_topics][texts_rate]:
                #     print(k)
                #     print(models[alpha][n_topics][texts_rate][k])
                #     print()
                # print("^^^^^^^^^^^^^")
                # print(alpha, n_topics, texts_rate)
                # print(emb[0].shape, emb[0].reshape((1, -1)).shape, type(emb[0]), models[alpha][n_topics][texts_rate]['model'].shape, type(models[alpha][n_topics][texts_rate]['model']))
                # aff = tm.compare(emb[0].reshape((1, -1)), models[alpha][n_topics][texts_rate]['model'])
                # print(aff)
                # print("^^^^^^^^^^^^^")

    df = pd.DataFrame(prosit_out, columns="method alpha texts_rate n_topics cv npmi weco rbo specif dissim descriptors".split())
    df.to_csv(f"{dirout}prosit_{method}.csv")
    print(df.loc[:, df.columns != 'descriptors'])
    return df


def runexp(texts_ll, texts_ls, vocab, count_csr, bow_corpus, embs, dirout):
    dirout_prosit_countbow = f"{dirout}prosit_countbow/"
    os.mkdir(dirout_prosit_countbow)
    df_all = compute_prosit(texts_ll, count_csr.toarray(), method='countbow', dirout=dirout_prosit_countbow)

    dirout_prosit_bertemb = f"{dirout}prosit_bertemb/"
    os.mkdir(dirout_prosit_bertemb)
    df = compute_prosit(texts_ll, embs, method='bertemb', dirout=dirout_prosit_bertemb)
    df_all = pd.concat([df_all, df], ignore_index=True)

    print(df_all.loc[:, df_all.columns != 'descriptors'])
    df_all.to_csv(f"{dirout}df_all.tsv", sep="\t")
    return df_all


def exp_jcr():
    texts_all = file2list(args.path, elemtype=str)
    texts_prep = texts_all[:args.limit]
    print(f"nr texts: {len(texts_all)}, nr texts: {len(texts_prep)}, under limit of {args.limit}")
    texts = [t.split() for t in texts_prep]
    dictionary = Dictionary(texts)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]
    csr_data = np.array([[freq, irow, icol] for irow, coldata in enumerate(bow_corpus) for (icol, freq) in coldata])
    count_csr = csr_matrix((csr_data[:, 0], (csr_data[:, 1], csr_data[:, 2])), shape=(len(texts), len(dictionary)))
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    embeddings = model.encode(texts_prep, show_progress_bar=False, batch_size=4)
    df_all  = runexp(texts, texts_prep, dictionary, count_csr, bow_corpus, embeddings, 'data/')
    return 1


exp_jcr()


