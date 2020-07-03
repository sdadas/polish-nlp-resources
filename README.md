# Polish NLP resources

This repository contains pre-trained models and language resources for Natural Language Processing in Polish created during my research work. If you'd like to use any of those resources in your research please cite:

```
@Misc{polish-nlp-resources,
  author =       {S{\l}awomir Dadas},
  title =        {A repository of Polish {NLP} resources},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/sdadas/polish-nlp-resources/}
}
```

## Contents

- [Word embeddings and language models](#word-embeddings-and-language-models)
- [Machine translation models](#machine-translation-models)
- [Dictionaries and lexicons](#dictionaries-and-lexicons)
- [Links to external resources](#links-to-external-resources)


## Word embeddings and language models

The following section includes pre-trained word embeddings and language models for Polish. Each model was trained on a corpus consisting of Polish Wikipedia dump, Polish books and articles, 1.5 billion tokens at total. 

### Word2Vec

Word2Vec trained with [Gensim](https://radimrehurek.com/gensim/). 100 dimensions, negative sampling, contains lemmatized words with 3 or more ocurrences in the corpus and additionally a set of pre-defined punctuation symbols, all numbers from 0 to 10'000, Polish forenames and lastnames. The archive contains embedding in gensim binary format. Example of usage:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load("word2vec_polish.bin")
    print(word2vec.similar_by_word("bierut"))
    
# [('cyrankiewicz', 0.818274736404419), ('gomułka', 0.7967918515205383), ('raczkiewicz', 0.7757788896560669), ('jaruzelski', 0.7737460732460022), ('pużak', 0.7667238712310791)]
```

[Download (Google Drive)](https://drive.google.com/open?id=1t2NsXHE0x5MfUvPR5MDV3_2TlxtdLkzz) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/word2vec.zip)

### FastText

FastText trained with [Gensim](https://radimrehurek.com/gensim/). Vocabulary and dimensionality is identical to Word2Vec model. The archive contains embedding in gensim binary format. Example of usage:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load("fasttext_100_3_polish.bin")
    print(word2vec.similar_by_word("bierut"))
    
# [('bieruty', 0.9290274381637573), ('gierut', 0.8921363353729248), ('bieruta', 0.8906412124633789), ('bierutow', 0.8795544505119324), ('bierutowsko', 0.839280366897583)]
```

[Download (Google Drive)](https://drive.google.com/open?id=1yfReM7EJGL1vk2dNbyM7X10I6k6lJMuX) (v2, trained with Gensim 3.8.0)

[Download (Google Drive)](https://drive.google.com/open?id=1_suJ-AxZ9yZ5zB5uW8UIaBJDNni83ZxJ) (v1, trained with Gensim 3.5.0, DEPRECATED)

### GloVe

Global Vectors for Word Representation (GloVe) trained using the reference implementation from Stanford NLP. 100 dimensions, contains lemmatized words with 3 or more ocurrences in the corpus. Example of usage:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load_word2vec_format("glove_100_3_polish.txt")
    print(word2vec.similar_by_word("bierut"))
    
# [('cyrankiewicz', 0.8335597515106201), ('gomułka', 0.7793121337890625), ('bieruta', 0.7118682861328125), ('jaruzelski', 0.6743760108947754), ('minc', 0.6692837476730347)]
```

[Download (Google Drive)](https://drive.google.com/open?id=1hLGZYOzG543p18ac-AfEsGXQGO6ioKex) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/glove.zip)

### High dimensional word vectors
Pre-trained vectors using the same vocabulary as above but with higher dimensionality. These vectors are more suitable for representing larger chunks of text such as sentences or documents using simple word aggregation methods (averaging, max pooling etc.) as more semantic information is preserved this way.

**GloVe** - **300d:** [Part 1 (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_300_3_polish.zip.001), **500d:** [Part 1 (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_500_3_polish.zip.001) [Part 2 (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_500_3_polish.zip.002), **800d:** [Part 1 (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_800_3_polish.zip.001) [Part 2 (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_800_3_polish.zip.002) [Part 3 (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_800_3_polish.zip.003) 

**Word2Vec** - [300d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EbNa5QXEYU5Jnbmq8gIK72YBRiQPybNBytVh2TaUCckyJQ?e=8Qa3vs), 
[500d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EQhO8-jVdWdHgVPSfDH-_0UB7N1PYCePdGAkt_y9Yuz0XA?e=NmrzhW), [800d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EepasgWhcIVLhACbANP4i-YBHaVgngZ0jJocsJNW0H80tw?e=YBk5bi)

**FastText** - [300d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/ERP19CsCaj1LoUe0ph3qujgBSIXGMBu4v92Hu-Yy9rJyag?e=8lpl5k), 
[500d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EX5ey8dihCdEkEFXeNrjzKsBmX9yeVPg92tdOsTfRmM_ug?e=NfwX9M), [800d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/Ebaa90Ft45ZAjvXGhAdrW2QBOJmfWMyl9czbwgYFqnZmvg?e=au9MLE)

### ELMo

Embeddings from Language Models (ELMo) is a contextual embedding presented in [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) by Peters et al. Sample usage with PyTorch below, for a more detailed instructions for integrating ELMo with your model please refer to the official repositories [github.com/allenai/bilm-tf](https://github.com/allenai/bilm-tf) (Tensorflow) and [github.com/allenai/allennlp](https://github.com/allenai/allennlp) (PyTorch).

```python
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder("options.json", "weights.hdf5")
print(elmo.embed_sentence(["Zażółcić", "gęślą", "jaźń"]))
```

[Download (Google Drive)](https://drive.google.com/open?id=110c2H7_fsBvVmGJy08FEkkyRiMOhInBP) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/elmo.zip)

### RoBERTa

Language model for Polish based on popular transformer architecture. We provide weights for improved BERT language model introduced in [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf). We provide two RoBERTa models for Polish - base and large model. A summary of pre-training parameters for each model is shown in the table below. We release two version of the each model: one in the [Fairseq](https://github.com/pytorch/fairseq) format and the other in the [HuggingFace Transformers](https://github.com/huggingface/transformers) format. More information about the models can be found in a [separate repository](https://github.com/sdadas/polish-roberta).

<table>
<thead>
<th>Model</th>
<th>L / H / A*</th>
<th>Batch size</th>
<th>Update steps</th>
<th>Corpus size</th>
<th>Final perplexity**</th>
<th>Fairseq</th>
<th>Transformers</th>
</thead>
<tr>
  <td>RoBERTa&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>125k</td>
  <td>~20GB</td>
  <td>3.66</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v2.9.0/roberta_base_transformers.zip">v2.9</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>30k</td>
  <td>50k</td>
  <td>~135GB</td>
  <td>2.92</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v2.9.0/roberta_large_transformers.zip">v2.9</a>
  </td>
</tr>
</table>

\* L - the number of encoder blocks, H - hidden size, A - the number of attention heads <br/>
\** Perplexity of the best checkpoint, computed on the validation split

Example in Fairseq:

```python
import os
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

model_path = "roberta_large_fairseq"
loaded = hub_utils.from_pretrained(
    model_name_or_path=model_path,
    data_name_or_path=model_path,
    bpe="sentencepiece",
    sentencepiece_vocab=os.path.join(model_path, "sentencepiece.bpe.model"),
    load_checkpoint_heads=True,
    archive_map=RobertaModel.hub_models(),
    cpu=True
)
roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
roberta.eval()
roberta.fill_mask('Druga wojna światowa zakończyła się w <mask> roku.', topk=1)
roberta.fill_mask('Ludzie najbardziej boją się <mask>.', topk=1)
#[('Druga wojna światowa zakończyła się w 1945 roku.', 0.9345270991325378, ' 1945')]
#[('Ludzie najbardziej boją się śmierci.', 0.14140743017196655, ' śmierci')]
```

It is recommended to use the above models, but it is still possible to download [our old model](https://github.com/sdadas/polish-nlp-resources/releases/download/roberta/roberta.zip), trained on smaller batch size (2K) and smaller corpus (15GB).

### Compressed Word2Vec

This is a compressed version of the Word2Vec embedding model described above. For compression, we used the method described in [Compressing Word Embeddings via Deep Compositional Code Learning](https://arxiv.org/abs/1711.01068) by Shu and Nakayama. Compressed embeddings are suited for deployment on storage-poor devices such as mobile phones. The model weights 38MB, only 4.4% size of the original Word2Vec embeddings. Although the authors of the article claimed that compressing with their method doesn't hurt model performance, we noticed a slight but acceptable drop of accuracy when using compressed version of embeddings. Sample decoder class with usage:

```python
import gzip
from typing import Dict, Callable
import numpy as np

class CompressedEmbedding(object):

    def __init__(self, vocab_path: str, embedding_path: str, to_lowercase: bool=True):
        self.vocab_path: str = vocab_path
        self.embedding_path: str = embedding_path
        self.to_lower: bool = to_lowercase
        self.vocab: Dict[str, int] = self.__load_vocab(vocab_path)
        embedding = np.load(embedding_path)
        self.codes: np.ndarray = embedding[embedding.files[0]]
        self.codebook: np.ndarray = embedding[embedding.files[1]]
        self.m = self.codes.shape[1]
        self.k = int(self.codebook.shape[0] / self.m)
        self.dim: int = self.codebook.shape[1]

    def __load_vocab(self, vocab_path: str) -> Dict[str, int]:
        open_func: Callable = gzip.open if vocab_path.endswith(".gz") else open
        with open_func(vocab_path, "rt", encoding="utf-8") as input_file:
            return {line.strip():idx for idx, line in enumerate(input_file)}

    def vocab_vector(self, word: str):
        if word == "<pad>": return np.zeros(self.dim)
        val: str = word.lower() if self.to_lower else word
        index: int = self.vocab.get(val, self.vocab["<unk>"])
        codes = self.codes[index]
        code_indices = np.array([idx * self.k + offset for idx, offset in enumerate(np.nditer(codes))])
        return np.sum(self.codebook[code_indices], axis=0)

if __name__ == '__main__':
    word2vec = CompressedEmbedding("word2vec_100_3.vocab.gz", "word2vec_100_3.compressed.npz")
    print(word2vec.vocab_vector("bierut"))
```

[Download (Google Drive)](https://drive.google.com/open?id=1vkAHM5m9AnWeVEaWqU2nXO_0Odkxsu49) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/compressed.zip)

## Machine translation models

This section includes pre-trained machine translation models.

### Polish-English and English-Polish convolutional models for Fairseq

We provide Polish-English and English-Polish convolutional neural machine translation models trained using [Fairseq](https://github.com/pytorch/fairseq) sequence modeling toolkit. Both models were trained on a parallel corpus of more than 40 million sentence pairs taken from [Opus](http://opus.nlpl.eu/) collection. Example of usage (`fairseq`, `sacremoses` and `subword-nmt` python packages are required to run this example):

```python
from fairseq.models import BaseFairseqModel

model_path = "/polish-english/"
model = BaseFairseqModel.from_pretrained(
    model_name_or_path=model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=model_path,
    tokenizer="moses",
    bpe="subword_nmt",
    bpe_codes="code",
    cpu=True
)
print(model.translate(sentence="Zespół astronomów odkrył w konstelacji Panny niezwykłą planetę.", beam=5))
# A team of astronomers discovered an extraordinary planet in the constellation of Virgo.
```

**Polish-English convolutional model:** [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/nmt-models-conv/polish-english-conv.zip) \
**English-Polish convolutional model:** [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/nmt-models-conv/english-polish-conv.zip)

## Dictionaries and lexicons

### Polish, English and foreign person names

This lexicon contains 346 thousand forenames and lastnames labeled as Polish, English or Foreign (other) crawled from multiple Internet sources.
Possible labels are: `P-N` (Polish forename), `P-L` (Polish lastname), `E-N` (English forename), `E-L` (English lastname), `F` (foreign / other). 
For each word, there is an additional flag indicating whether this name is also used as a common word in Polish (`C` for common, `U` for uncommon).

[Download (GitHub)](lexicons/names)

### Named entities extracted from SJP.PL

This dictionary consists mostly of the names of settlements, geographical regions, countries, continents and words derived from them (relational adjectives and inhabitant names). 
Besides that, it also contains names of popular brands, companies and common abbreviations of institutions' names.
This resource was created in a semi-automatic way, by extracting the words and their forms from SJP.PL using a set of heuristic rules and then manually filtering out words that weren't named entities.

[Download (GitHub)](lexicons/named-sjp)

## Links to external resources

### Repositories of linguistic tools and resources

- [Computational Linguistics in Poland - IPI PAN](http://clip.ipipan.waw.pl/LRT)
- [G4.19 Research Group, Wroclaw University of Technology](http://nlp.pwr.wroc.pl/narzedzia-i-zasoby)
- [CLARIN - repository of linguistic resources](https://clarin-pl.eu/dspace/)
- [Gonito.net - evaluation platform with some challenges for Polish](https://gonito.net)

### Polish models

- [Marian-NMT](https://marian-nmt.github.io/) - An efficient C++ based implementation of neural translation models. Many pre-trained models are available, including those supporting Polish: [pl-de](https://huggingface.co/Helsinki-NLP/opus-mt-pl-de), [pl-en](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en), [pl-es](https://huggingface.co/Helsinki-NLP/opus-mt-pl-es), [pl-fr](https://huggingface.co/Helsinki-NLP/opus-mt-pl-fr), [pl-sv](https://huggingface.co/Helsinki-NLP/opus-mt-pl-sv), [de-pl](https://huggingface.co/Helsinki-NLP/opus-mt-de-pl), [es-pl](https://huggingface.co/Helsinki-NLP/opus-mt-es-pl), [fr-pl](https://huggingface.co/Helsinki-NLP/opus-mt-fr-pl).

### Multilingual models supporting Polish language

- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) - BERT (Bidirectional Encoder Representations from Transformers) is a model for generating contextual word representations. Multilingual cased model provided by Google supports 104 languages including Polish.
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1) - USE (Universal Sentence Encoder) generates sentence level langauge representations. Pre-trained multilingual model supports 16 langauges (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian).
- [LASER Language-Agnostic SEntence Representations](https://github.com/facebookresearch/LASER) - A multilingual sentence encoder by Facebook Research, supporting 93 languages.
- [XLM-RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) - Cross lingual sentence encoder trained on 2.5 terabytes of data from CommonCrawl and Wikipedia. Supports 100 languages including Polish. See [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf) for details.
- [Slavic BERT](https://github.com/deepmipt/Slavic-BERT-NER#slavic-bert) - Multilingual BERT model supporting Bulgarian (bg), Czech (cs), Polish (pl) and Russian (ru) languages.
