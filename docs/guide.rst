Guide
=====

.. image:: https://img.shields.io/pypi/v/boostsa.svg
        :target: https://pypi.python.org/pypi/prosit

.. image:: https://img.shields.io/github/license/fornaciari/boostsa
        :target: https://lbesson.mit-license.org/
        :alt: License

.. image:: https://github.com/fornaciari/boostsa/workflows/Python%20Package/badge.svg
        :target: https://github.com/fornaciari/prosit/actions

.. image:: https://readthedocs.org/projects/boostsa/badge/?version=latest
        :target: https://boostsa.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1eewGMqW_cIRqKdWW1tBCFE3T2qVCI_EV#scrollTo=6czDoYOiGpJx
    :alt: Open In Colab

Intro
-----

ProSiT - PROgressive SImilarity Thresholds is an algorithm for topic models.
Given a corpus of texts, it will find latent dimensions corresponding
to the main topics present in the corpus, providing for each of them the relative keywords (descriptors).

It is input agnostic: it can deal with any kind of textual representations, be they vectors resulting from, for example,
Bag of Words - BoW or State Of The Art - SOTA (multi-lingual) Language Models - LMs.

ProSiT is deterministic and fully interpretable.

It does not require any assumption regarding the possible number of topics in a corpus of documents:
they are automatically identified given two tunable similarity parameters, :math:`\alpha` and :math:`\beta`.

The :math:`\alpha` parameter is used to determine the minimum Cosine Similarity Threshold - CST to consider different documents
as related to the same latent dimension, i.e. topic.
PRoSiT is an iterative algorithm, that finds the latent dimensions in different epochs, that need progressively higher similarity thresholds.
The :math:`\alpha` parameter is used in the following formula:

.. math::

    CST = \frac{iter - \alpha}{iter}

This produce this kind of curves,:

.. image:: _static/alpha.png
    :width: 500
    :alt: Some examples of the :math:`\alpha` curve.










