# coding=latin-1
# Non nobis, Domine, non nobis, sed Nomini Tuo da gloriam
import sys, os, re, time, json
import numpy as np
import pandas as pd
import pickle5 as pickle
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix, save_npz
import torch
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO, CoherenceWordEmbeddings
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sb
import imageio
from random import sample
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################


class PST:
    def __init__(self, alpha=.001, max_topics=50, min_topics=4, nr_descriptors=10, closest_texts_rate=.1, verbose=False, device='cuda:0', dtype=torch.float32, dir_out=''):
        self.alphas = alpha if type(alpha) is list else [alpha]
        self.max_topics = max_topics
        self.min_topics = min_topics
        self.closest_texts_rates = closest_texts_rate if type(closest_texts_rate) is list else [closest_texts_rate]
        self.nr_descriptors = nr_descriptors
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        self.dirout = dir_out + '/' if (dir_out != '') and (not re.search('/$', dir_out)) else dir_out
        if not os.path.exists(self.dirout): os.mkdir(self.dirout)

    @staticmethod
    def list2file(lis, fileout, sepline="\n", wra='w'):
        with open(fileout, wra) as fileout: [fileout.write(str(x) + sepline) for x in lis]
        return 1

    @staticmethod
    def start(sep=True):
        start = time.time()
        now = time.strftime("%Y/%m/%d %H:%M:%S")
        if sep: print('#' * 80)
        print('start:', now)
        return start

    @staticmethod
    def end(start, sep=True):
        end = time.time()
        dur = end - start

        def stringtime(n):
            h = str(int(n / 3600))
            m = str(int((n % 3600) / 60))
            s = str(int((n % 3600) % 60))
            if len(h) == 1: h = '0' + h
            if len(m) == 1: m = '0' + m
            if len(s) == 1: s = '0' + s
            return h + ':' + m + ':' + s

        str_dur = stringtime(end - start)
        now = time.strftime("%Y/%m/%d %H:%M:%S")
        if sep:
            print('#' * 80 + "\nend:", now, " - time elapsed:", str_dur + "\n" + '#' * 80)
        else:
            print("end:", now, " - time elapsed:", str_dur)
        return dur

    @staticmethod
    def writejson(data, pathname, wra='w', indent=2):
        with open(pathname, wra) as out: json.dump(data, out, indent=indent)
        return 1

    @staticmethod
    def writebin(data, f_out):
        out = open(f_out, "wb")
        pickle.dump(data, out)
        out.close()
        return 1

    @staticmethod
    def linscale(M, min_wanted=0, max_wanted=1, axis=0, dec=4):
        if len(M.shape) == 1:
            max_number = M.max()
            min_number = M.min()
            scale = (max_wanted - min_wanted) / (max_number - min_number)
            offset = ((max_number * min_wanted) - (min_number * max_wanted)) / (max_number - min_number)
            M = ((M * scale) + offset).round(dec)
        elif axis == 0:
            for i in range(M.shape[0]):
                max_number = M[i, :].max()
                min_number = M[i, :].min()
                scale = (max_wanted - min_wanted) / (max_number - min_number)
                offset = ((max_number * min_wanted) - (min_number * max_wanted)) / (max_number - min_number)
                M[i, :] = ((M[i, :] * scale) + offset).round(dec)
        elif axis == 1:
            for i in range(M.shape[1]):
                max_number = M[:, i].max()
                min_number = M[:, i].min()
                scale = (max_wanted - min_wanted) / (max_number - min_number)
                offset = ((max_number * min_wanted) - (min_number * max_wanted)) / (max_number - min_number)
                M[:, i] = ((M[:, i] * scale) + offset).round(dec)
        return M
    
    def makepng(self, Z_pos, Z_col, dims=3, pathout='out.png'):
        sb.set_context('poster')
        fig = plt.figure(figsize=(12, 8))
        if dims == 2:
            ax = fig.add_subplot(111)
            ax.scatter(Z_pos[:, 0], Z_pos[:, 1], c = Z_col)
        elif dims == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Z_pos[:, 0], Z_pos[:, 1], Z_pos[:, 2], c=Z_col)
        # plt.title(f"{X.shape[0]} data points", fontsize=40)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        # ax.set_xlim((min(Z_pos[:, 0]), max(Z_pos[:, 0])))
        # ax.set_ylim((min(Z_pos[:, 1]), max(Z_pos[:, 1])))
        # ax.set_zlim((min(Z_pos[:, 2]), max(Z_pos[:, 2])))
        plt.tight_layout()
        plt.savefig(pathout, dpi=300)
        # plt.show()
        plt.close()
        return None

    def specificity(self, p, q):
        """
        p -> topics -> [voc, n_topics]
        q -> corpus -> [voc, 1];
        kl_divergences -> [1, n_topics];
        """
        p = np.where(p != 0, p, 0.0000001)
        q = np.where(q != 0, q, 0.0000001)
        kl_divergences = np.sum(p * np.log(p / q), axis=0)
        return np.mean(kl_divergences)

    def evaluate(self, texts, dictionary, descriptors, cv=True, npmi=True, weco=True, rbo=True, distinct=True):
        if self.verbose: print(f"computing metrics:")
        metric2value = defaultdict(lambda: 'not computed')
        if cv:
            try:
                coherence_cv = CoherenceModel(texts=texts, dictionary=dictionary, topics=descriptors, coherence='c_v', processes=1)
                if self.verbose: print(f"cv:   {coherence_cv.get_coherence():.4f}")
                metric2value['cv'] = round(coherence_cv.get_coherence(), 4)
            except Exception as err:
                metric2value['cv'] = err
        if npmi:
            try:
                coherence_npmi = CoherenceModel(texts=texts, dictionary=dictionary, topics=descriptors, coherence='c_npmi', processes=1)
                if self.verbose: print(f"npmi: {coherence_npmi.get_coherence():.4f}")
                metric2value['npmi'] = round(coherence_npmi.get_coherence(), 4)
            except Exception as err:
                metric2value['npmi'] = err
        if weco:
            try:
                coherence_weco = CoherenceWordEmbeddings(descriptors).score()
                if self.verbose: print(f"weco: {coherence_weco:.4f}")
                metric2value['weco'] = round(coherence_weco, 4)
            except Exception as err:
                metric2value['weco'] = err
        if rbo:
            try:
                coherence_rbo = InvertedRBO(descriptors).score()
                if self.verbose: print(f"rbo:  {rbo:.4f}")
                metric2value['rbo'] = round(coherence_rbo, 4)
            except Exception as err:
                metric2value['rbo'] = err
        if distinct:
            try:
                w2f = Counter([word for topic in descriptors for word in topic])
                w2i = {w: i for i, w in enumerate(w2f.keys())}
                # wordfreq4topic = np.zeros([len(w2f), len(descriptors)])
                wordfreq4topic = lil_matrix((len(w2f), len(descriptors)), dtype=int)
                for itopic, topic in enumerate(descriptors):
                    for word in topic:
                        wordfreq4topic[w2i[word], itopic] += 1
                wordfreq4topic = wordfreq4topic.tocsr()
                corpus_distribution = np.sum(wordfreq4topic, axis=1) / np.sum(wordfreq4topic)
                topics_distribution = wordfreq4topic / np.sum(wordfreq4topic, axis=0)
                specif = self.specificity(topics_distribution, corpus_distribution)
                dissim = max([np.linalg.norm(np.mean(topics_distribution[:, i] - np.delete(topics_distribution, i, axis=1), axis=1), 2) for i in range(topics_distribution.shape[1])])
                metric2value['specif'] = round(specif, 4)
                metric2value['dissim'] = round(dissim, 4)
                if self.verbose:
                    print(f"specif:  {specif:.4f}\ndissim:  {dissim:.4f}")
            except Exception as err:
                metric2value['specif'] = err
                metric2value['dissim'] = err
        return metric2value

    def compute_topterms(self, doc4terms_matrix, dictionary, labels):
        labsize = len(set(labels))
        descriptors = list()
        for ilab in range(labsize):
            m = doc4terms_matrix[labels == ilab]
            scores = np.sum(m, axis=0)
            itopterms = np.argsort(scores)[::-1]
            topic_descriptors = [dictionary[idx] for idx in itopterms[:self.nr_descriptors]]
            descriptors.append(topic_descriptors)
            print(f"class {ilab+1}: {topic_descriptors}")
        return descriptors
        
    @staticmethod
    def rank(l):
        sort_idx_val = sorted([(i, v) for i, v in enumerate(l)], key=lambda x: x[1])
        sort_rank_idx = sorted([(i, sort_idx_val[i][0]) for i in range(len(sort_idx_val))], key=lambda x: x[1])
        return [tup[0] for tup in sort_rank_idx]

    @staticmethod
    def entropy(x, y):
        if x == 0 or y == 0:
            return 0  # l'altro termine va a 1, che il log porta a 0
        else:
            return -((x / (x + y)) * np.log2(x / (x + y))) - ((y / (x + y)) * np.log2(y / (x + y)))

    def compute_ig(self, texts, labels, name=''):
        labsize = len(set(labels))
        txtsize = len(texts)
        voc = sorted({w for txt in texts for w in txt}) # ordinare le parole in modo costante è fondamentale per le misure, e quindi per replicare le misure di performance
        vocsize = len(voc)
        if self.verbose: print(f"computing ig {name}...\nvocabulary size: {vocsize}")
        w2i = {w: i for i, w in enumerate(voc)}
        wordfreq4topic = lil_matrix((vocsize, labsize), dtype=int)
        for irow in range(len(texts)):
            for w in set(texts[irow]):
                wordfreq4topic[w2i[w], labels[irow]] += 1
        wordfreq4topic = wordfreq4topic.tocsr()
        label2freq = Counter(labels)
        word2metrics = defaultdict(dict)#defaultdict(lambda: {'ig': 0, 'igr': 0, 'freq': 0, 'frequent_class': 2})
        # for w, iw in sorted(w2i.items(), key=lambda x: x[1]):
        for w in w2i:
            iw = w2i[w]
            word_distribution = wordfreq4topic.toarray()[iw, :]
            nr_found = sum(word_distribution)
            p_word = nr_found / txtsize  # forman: p_word = (tp + fp) / all
            p_not_word = 1 - p_word
            ig = 0
            probs_tp = list()
            for label in sorted(label2freq): # sorted necessario per mantenere l'ordine delle labels allineato con la successiva lista topics, che crea le labels con ilab
                pos = label2freq[label]
                neg = txtsize - pos
                tp = word_distribution[label]
                fp = nr_found - tp
                tn = neg - fp
                fn = pos - tp
                ig += self.entropy(pos, neg) - ((p_word * self.entropy(tp, fp)) + (p_not_word * self.entropy(fn, tn)))
                p_tp = tp/nr_found
                # word2metrics[w][f"p{label}"] = p_tp
                probs_tp.append(p_tp)

            # ie = - (p_word * np.log2(p_word)) - (p_not_word * np.log2(p_not_word))  # intrinsec entropy, Han, Geolocation Prediction in..., sez 4.2
            word2metrics[w]['ig'] = round(ig, 6)
            # word2metrics[w]['igr'] = round(ig / ie, 6)
            word2metrics[w]['freq'] = nr_found
            for i, p in enumerate(probs_tp): word2metrics[w][f"prob{i}"] = p
            for i, r in enumerate(self.rank(probs_tp)): word2metrics[w][f"rank{i}"] = r
            
        df_all = pd.DataFrame([{k: word2metrics[w][k] for k in word2metrics[w]} for w in word2metrics], index=[w for w in word2metrics])
        df_all.sort_values(by=['ig'], ascending=False, inplace=True)

        # df_all.to_excel(f"{self.dirout}word2metrics{'_' if name != '' else ''}{name}.xlsx")
        # self.writejson(word2metrics, f"{dirout}word2metrics{'_' if name != '' else ''}{name}.json")
        # print(df_all.head(10))
        topics = list()
        for ilab in range(labsize):
            go = self.nr_descriptors
            bestrank = labsize - 1
            descriptors = list()
            while go > 0:
                df = df_all[df_all[f"rank{ilab}"] == bestrank][:self.nr_descriptors]
                descriptors.extend(df.index.tolist())
                bestrank -= 1
                go -= len(descriptors)
            descriptors = descriptors[:self.nr_descriptors]
            print(f"class {ilab+1}: {descriptors}")
            topics.append(descriptors)
        return topics, df_all

    @staticmethod
    def cosine_similarity_matrix(m1, m2):
        return torch.mm(m1, m2) / torch.mm(torch.linalg.norm(m1, dim=1).unsqueeze(1), torch.linalg.norm(m2.t(), dim=1).unsqueeze(0))

    def fit(self, instance_texts, instance_vectors, cv=True, npmi=True, weco=True, rbo=True, distinct=True, png=False, pngdims=2, pngpoints=1000):
        assert len(instance_texts) == len(instance_texts), f"nr texts ({len(instance_texts)}) must be equal to nr vectors ({len(instance_texts)})"
        instance_vectors = instance_vectors.to(device=self.device, dtype=self.dtype) if torch.is_tensor(instance_vectors) else torch.from_numpy(instance_vectors).to(device=self.device, dtype=self.dtype)
        models = dict()
        startime = self.start()
        for alpha in self.alphas:
            models[alpha] = dict()
            inputvecs = {tuple(vector.tolist()) for vector in instance_vectors}
            cluster_vectors = torch.stack([torch.tensor(tup) for tup in inputvecs]).to(device=self.device, dtype=self.dtype)
            epoch = 0
            png_epoch_bounds = [0]
            if png:
                if cluster_vectors.shape[0] > pngpoints:
                    idxs = list(range(cluster_vectors.shape[0]))
                    sample_idxs = sample(idxs, pngpoints)
                    X_png = cluster_vectors.cpu().numpy()[sample_idxs]
                    png_epoch_bounds.append(png_epoch_bounds[-1] + cluster_vectors.cpu().numpy()[sample_idxs].shape[0])
                else:
                    X_png = cluster_vectors.cpu().numpy()
                    png_epoch_bounds.append(png_epoch_bounds[-1] + cluster_vectors.cpu().numpy().shape[0])
            while len(cluster_vectors) > self.min_topics:
                epoch += 1
                simatrix = self.cosine_similarity_matrix(cluster_vectors, cluster_vectors.t())
                """
                Setting the threshold of minimum similarity for the epoch.
                At every epoch, the threshold asymptotically tends to 1, to prevent the centroid to go too far away from their initial position, and too close to the overall data set centroid
                """
                clusim_thres = (epoch - alpha) / epoch
                # print(f"{'#'*70}\nepoch {epoch}\n{'alpha ' + str(alpha) + ' ':.<40} {clusim_thres:.8f}\n{'clusters shape':.<40} {cluster_vectors.shape}")
                print(f"epoch {epoch:<3}- alpha {alpha:<7}- threshold {clusim_thres:<12.8f}- clusters shape begin {cluster_vectors.shape}")
                """
                Setting to 1 the similarity matrix cells with value > the similarity threshold
                torch.nonzero, with as_tuple=True, returns 2 tuples, containing the row and column indexes, respectively, where the input is nonzero
                """
                edge = torch.nonzero((simatrix > clusim_thres), as_tuple=True)
                irows, icols = edge
                """
                Creating a dict:
                Key: cluster index
                Value: set of neighbor cluster indexes, including itself
                """
                ivec2ineighs = defaultdict(set)
                for irow, icol in zip(irows.tolist(), icols.tolist()):
                # for irow, icol in tqdm(zip(irows.tolist(), icols.tolist()), "collecting neighbors"):
                    ivec2ineighs[irow].add(icol)
                """
                Creating a set of sets, sorted by length, from the shortest.
                Each set contains the vector indexes of a cluster
                the frozenset automatically discards the duplicated sets.
                """
                clusters = sorted({frozenset(ineigh) for ineigh in ivec2ineighs.values()}, key=len)
                if self.verbose: print(f"{'nr clusters within min sim':.<40} {len(clusters)}")
                """
                Removing subsets of sets, in two steps.
                First, collecting the indexes of the subsets:
                For each outer_index:
                    for each inner_index, from outer_index + 1:
                        if the cluster[outer_index] is contained in the cluster[inner_index]
                            I collect the outer_index and skip to the next outer loop
                """
                indices_to_delete = list()
                for iout in range(len(clusters)):
                # for iout in tqdm(range(len(clusters)), "removing subsets"):
                    # for iin in range(len(clusters) - 1, iout, -1):
                    for iin in range(iout + 1, len(clusters)):
                        if clusters[iout].issubset(clusters[iin]):
                            indices_to_delete.append(iout)
                            break # "no way to break a list comprehension"
                """Second, selecting only the clusters that are not subsets:"""
                clusters = [clusters[i] for i in range(len(clusters)) if i not in indices_to_delete]
                if self.verbose: print(f"{'nr clusters without subsets':.<40} {len(clusters)}")
                if len(clusters) <= self.min_topics:
                    # print(f"no points within the similarity threshold, loop ended")
                    break
                """
                Creating a dict:
                Key: the new cluster index (from enumerate(clusters))
                Value: a tuple of two elements:
                    - the indexes of the cluster vectors
                    - the mean of those vectors
                """
                iclu2ivecs_centroid = {iclu: (cluster, cluster_vectors[list(cluster)].mean(dim=0)) for iclu, cluster in enumerate(clusters)}
                """Creating a tensor containing the clusters centroids"""
                centroids = torch.stack([iclu2ivecs_centroid[iclu][1] for iclu in iclu2ivecs_centroid])
                """
                Assigning the clusters constituted by a single vector to the cluster with the closest centroid
                For each cluster:
                    if it contains only one index vector
                        I compute the similarity of that vector with the clusters centroids
                        I select the index of the cluster with max similarity (excluding itself)
                        I add the lonely index to that cluster
                """
                for iclu in iclu2ivecs_centroid:
                # for iclu in tqdm(iclu2ivecs_centroid, 'assigning single cases'):
                    if len(iclu2ivecs_centroid[iclu][0]) == 1:
                        isinglevec = list(iclu2ivecs_centroid[iclu][0])[0]
                        singlevec = iclu2ivecs_centroid[iclu][1]
                        sims = self.cosine_similarity_matrix(centroids, singlevec.unsqueeze(1)).squeeze(1).tolist()
                        iclu_maxsim = max([(i, s) for i, s in enumerate(sims) if i != iclu], key=lambda x: x[1])[0]
                        new_cluster = set(clusters[iclu_maxsim])
                        new_cluster.add(isinglevec)
                        clusters[iclu_maxsim] = frozenset(new_cluster)
                """Selecting only the clusters consititued by more than one vector"""
                clusters = [clu for clu in clusters if len(clu) > 1]
                """Recomputing the centroids"""
                iclu2ivecs_centroid = {iclu: (cluster, cluster_vectors[list(cluster)].mean(dim=0)) for iclu, cluster in enumerate(clusters)}
                """Updating the tensor containing the cluster centroids"""
                cluster_vectors = torch.stack([iclu2ivecs_centroid[iclu][1] for iclu in iclu2ivecs_centroid]).to(device=self.device, dtype=self.dtype)
                if self.verbose: print(f"{'clusters shape':.<40} {cluster_vectors.shape}")
                """Removing the duplicated centroids: it happens"""
                uniquevecs = {tuple(vector.tolist()) for vector in cluster_vectors}
                cluster_vectors = torch.stack([torch.tensor(tup) for tup in uniquevecs]).to(device=self.device, dtype=self.dtype)
                if self.verbose: print(f"{'clusters shape without duplicates':.<40} {cluster_vectors.shape}")
                """If the number of clusters is <= than the max nr of topics threshold, I compute the topics' features"""
                if png and len(cluster_vectors) >= self.min_topics:
                    if cluster_vectors.shape[0] > pngpoints:
                        idxs = list(range(cluster_vectors.shape[0]))
                        sample_idxs = sample(idxs, pngpoints)
                        X_png = np.append(X_png, cluster_vectors.cpu().numpy()[sample_idxs], 0)
                        png_epoch_bounds.append(png_epoch_bounds[-1] + cluster_vectors.cpu().numpy()[sample_idxs].shape[0])
                    else:
                        # self.makepng(cluster_vectors.cpu().numpy(), dims=pngdims, pathout=f"{self.dirout}alpha{alpha}epo{epoch}.png")
                        X_png = np.append(X_png, cluster_vectors.cpu().numpy(), 0)
                        png_epoch_bounds.append(png_epoch_bounds[-1] + cluster_vectors.cpu().numpy().shape[0])
                    # images.append(imageio.imread(f"{self.dirout}alpha{alpha}epo{epoch}.png"))
                if self.max_topics >= len(cluster_vectors) >= self.min_topics:
                    models[alpha][len(cluster_vectors)] = dict()
                    """
                    Computing the similarity matrix between the out-of-while, original docs representations and the clusters' centroids.
                    At the end of the process, for each doc, the closest topic is not necessarily the same that the doc contributed to create,
                    that is the "vec" for the "ivecs" in inputvec2iinputvecs, that is redefined by centroid2ivecs at each iteration.
                    """
                    doc4topic_cosim = self.cosine_similarity_matrix(instance_vectors, cluster_vectors.t())
                    topics = np.array([np.argmax(row) for row in doc4topic_cosim.cpu().numpy()])
                    """
                    Può capitare che dopo argmax nessun testo sia vicino a qualche topic
                    devo rimuovere le colonne da cluster_vectors e modificare gli indici successivi a quelli venuti a mancare
                    In pratica faccio questo:
                    a = [1,1,3,3,5,5]
                    sa = sorted(set(a))
                    ra = list(range(6))
                    {s:r for s,r in zip(sa, ra)}
                    output: {1: 0, 3: 1, 5: 2}
                    """
                    cluster_vectors = cluster_vectors[[i for i in range(len(cluster_vectors)) if i in set(topics)], :]
                    found2newidx = {found: newidx for found, newidx in zip(sorted(set(topics)), list(range(doc4topic_cosim.shape[1])))}
                    topics = np.array([found2newidx[t] for t in topics])
                    """Counting the topics"""
                    top2freq = Counter(topics)
                    for c, n in top2freq.most_common(): print(f"{n:<5} instances in topic {c+1}")
                    print(f"{'final nr clusters':.<40} {len(cluster_vectors)}")
                    """Converting directly to np.array the texts, if they are a list of list of words; if they are a list of string, i do the same, converting the string to lists of words"""
                    instance_texts = np.array(instance_texts, dtype=object) if isinstance(instance_texts[0], list) else np.array([row.split() for row in instance_texts], dtype=object) if isinstance(instance_texts[0], str) else sys.exit('texts must be a list of string or a list of lists')
                    for closest_texts_rate in self.closest_texts_rates:
                        print(f"closest_texts_rate: {closest_texts_rate}")
                        """
                        Creating the list of indexes of the most representative texts for each topic
                        ci sono testi che sono tra i più vicini di diversi cluster. con closest_texts_rate molto basso, può succedere che un topic rimanga senza doc per i descriptors
                        uso seen, in modo da non sovrascrivere il topic sugli indici già visti.
                        se un indice è unseen, in dalla lista topics riceve comunque il topic a cui è più vicino
                        """
                        i_representative_texts = list()
                        df_aff = pd.DataFrame(doc4topic_cosim.cpu().numpy())
                        seen = set() #
                        for itop in range(doc4topic_cosim.shape[1]):
                            first_docs = int(top2freq[itop] * closest_texts_rate)
                            idocs = df_aff.sort_values(by=[itop], ascending=False).index[:first_docs].tolist()
                            idocs_selected = set(idocs).difference(seen)
                            # for idoc in idocs_selected:
                            #     if topics[idoc] != itop:
                            #         print(f"topic {itop}: doc index {idoc} delected for topic {topics[idoc]}")
                            i_representative_texts.extend(idocs_selected)
                            seen.update(idocs)
                        
                        missing_topics = set(topics).difference(set(topics[i_representative_texts]))
                        if bool(missing_topics):
                            print(f"*******\nTopic {missing_topics}: no docs left for descriptors\nDescriptors not computed!\nPlease set a greater closest_texts_rate.\n*******")
                            continue
                        
                        descriptors, df_words = self.compute_ig(instance_texts[i_representative_texts], topics[i_representative_texts])
                        dictionary = Dictionary(instance_texts)
                        metric2value = self.evaluate(instance_texts, dictionary, descriptors, cv=cv, npmi=npmi, weco=weco, rbo=rbo, distinct=distinct)
                        output_vectors = cluster_vectors.cpu().numpy()
                        models[alpha][len(cluster_vectors)][closest_texts_rate] = {'model': output_vectors,
                                                                                   'doc4topicaffinity': doc4topic_cosim,
                                                                                   'doctopics': topics,
                                                                                   'descriptors': descriptors,
                                                                                   'df_words': df_words,
                                                                                   'cv': metric2value['cv'],
                                                                                   'npmi': metric2value['npmi'],
                                                                                   'weco': metric2value['weco'],
                                                                                   'rbo': metric2value['rbo'],
                                                                                   'specif': metric2value['specif'],
                                                                                   'dissim': metric2value['dissim']}
            if png:
                images = list()
                svd_pos = TruncatedSVD(n_components=pngdims, n_iter=10, algorithm='randomized', random_state=42)
                svd_col = TruncatedSVD(n_components=3, n_iter=10, algorithm='randomized', random_state=42)
                Z_pos = self.linscale(svd_pos.fit_transform(X_png), -1, 1, axis=1)
                Z_col = self.linscale(svd_col.fit_transform(X_png), 0, 1, axis=1)
                for iprev in range(len(png_epoch_bounds) - 1):
                    inext = iprev + 1
                    self.makepng(Z_pos[png_epoch_bounds[iprev]: png_epoch_bounds[inext], :], Z_col[png_epoch_bounds[iprev]: png_epoch_bounds[inext], :], dims=pngdims, pathout=f"{self.dirout}alpha{alpha}epo{inext}.png")
                    images.append(imageio.imread(f"{self.dirout}alpha{alpha}epo{inext}.png"))
                imageio.mimsave(f"{self.dirout}alpha{alpha}.gif", images, format='GIF', duration=1)
        self.writebin(models, f"{self.dirout}models.bin")
        self.end(startime)
        return models

    def compare(self, docs, topics):
        docs = docs.to(device=self.device, dtype=self.dtype) if torch.is_tensor(docs) else torch.from_numpy(docs).to(device=self.device, dtype=self.dtype)
        topics = topics.to(device=self.device, dtype=self.dtype) if torch.is_tensor(topics) else torch.from_numpy(topics).to(device=self.device, dtype=self.dtype)
        return self.cosine_similarity_matrix(docs, topics.t()).numpy()