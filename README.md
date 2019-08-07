# Polish NLP resources

This repository contains pre-trained models and language resources for Natural Language Processing in Polish created during my research work, including:

*Sławomir Dadas: “Combining neural and knowledge-based approaches to Named Entity Recognition in Polish”, 2018; [arXiv:1811.10418](https://arxiv.org/abs/1811.10418).*

## Word embeddings

The following section includes pre-trained word embedding models for Polish. Each model was trained on a corpus consisting of Polish Wikipedia dump, Polish books and articles, 1.5 billion tokens at total. 

### Word2Vec

Word2Vec trained with [Gensim](https://radimrehurek.com/gensim/). 100 dimensions, negative sampling, contains lemmatized words with 3 or more ocurrences in the corpus and additionally a set of pre-defined punctuation symbols, all numbers from 0 to 10'000, Polish forenames and lastnames. The archive contains embedding in gensim binary format. Sample usage:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load("word2vec_polish.bin")
    print(word2vec.similar_by_word("bierut"))
    
# [('cyrankiewicz', 0.818274736404419), ('gomułka', 0.7967918515205383), ('raczkiewicz', 0.7757788896560669), ('jaruzelski', 0.7737460732460022), ('pużak', 0.7667238712310791)]
```

[Download (Google Drive)](https://drive.google.com/open?id=1t2NsXHE0x5MfUvPR5MDV3_2TlxtdLkzz) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/word2vec.zip)

**Warning!** For newer versions of Gensim, try renaming the file `word2vec_100_3_polish.npy` to `word2vec_100_3_polish.bin.syn0.npy` if you encounter any problems loading the embeddings.

### FastText

FastText trained with [Gensim](https://radimrehurek.com/gensim/). Vocabulary and dimensionality is identical to Word2Vec model. The archive contains embedding in gensim binary format. Sample usage:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load("fasttext_100_3_polish.bin")
    print(word2vec.similar_by_word("bierut"))
    
# [('bieruty', 0.9290274381637573), ('gierut', 0.8921363353729248), ('bieruta', 0.8906412124633789), ('bierutow', 0.8795544505119324), ('bierutowsko', 0.839280366897583)]
```

[Download (Google Drive)](https://drive.google.com/open?id=1yfReM7EJGL1vk2dNbyM7X10I6k6lJMuX) (v2, trained on Gensim 3.8.0)

[Download (Google Drive)](https://drive.google.com/open?id=1_suJ-AxZ9yZ5zB5uW8UIaBJDNni83ZxJ) (v1, trained on Gensim 3.5.0, DEPRECATED)

### GloVe

Global Vectors for Word Representation (GloVe) trained using the reference implementation from Stanford NLP. 100 dimensions, contains lemmatized words with 3 or more ocurrences in the corpus. Sample usage:

```python
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word2vec = KeyedVectors.load_word2vec_format("glove_100_3_polish.txt")
    print(word2vec.similar_by_word("bierut"))
    
# [('cyrankiewicz', 0.8335597515106201), ('gomułka', 0.7793121337890625), ('bieruta', 0.7118682861328125), ('jaruzelski', 0.6743760108947754), ('minc', 0.6692837476730347)]
```

[Download (Google Drive)](https://drive.google.com/open?id=1hLGZYOzG543p18ac-AfEsGXQGO6ioKex) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/glove.zip)
 
### ELMo

Embeddings from Language Models (ELMo) is a contextual embedding presented in [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) by Peters et al. Since the usage of this model is not trivial and there are several different ways of integrating it into deep learning architecture, for more information on ELMo please refer to the official repositories [github.com/allenai/bilm-tf](https://github.com/allenai/bilm-tf) (Tensorflow) and [github.com/allenai/allennlp](https://github.com/allenai/allennlp) (PyTorch).

[Download (Google Drive)](https://drive.google.com/open?id=110c2H7_fsBvVmGJy08FEkkyRiMOhInBP) or [Download (GitHub)](https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/elmo.zip)

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
