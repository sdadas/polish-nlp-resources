# Polish NLP resources

This repository contains pre-trained models and language resources for Natural Language Processing in Polish created during my research. Some of the models are also available on **[Huggingface Hub](https://huggingface.co/sdadas)**.

If you'd like to use any of those resources in your research please cite:

```bibtex
@Misc{polish-nlp-resources,
  author =       {S{\l}awomir Dadas},
  title =        {A repository of Polish {NLP} resources},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/sdadas/polish-nlp-resources/}
}
```

## Contents

- [Word embeddings](#word-embeddings)
- [Language models](#language-models)
- [Sentence encoders](#sentence-encoders)
- [Machine translation models](#machine-translation-models)
- [Dictionaries and lexicons](#dictionaries-and-lexicons)
- [Links to external resources](#links-to-external-resources)


## Word embeddings

The following section includes pre-trained word embeddings for Polish. Each model was trained on a corpus consisting of Polish Wikipedia dump, Polish books and articles, 1.5 billion tokens at total. 

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

**Word2Vec** - [300d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EQ7QA6PkPupBtZYyP8kaafMB0z9FdHfME7kxm_tcRWh9hA?e=RGekMu), 
[500d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EfBT7WvY7eVHuQZIuUpnJzsBXTN2L896ldVvhRBiCmUH_A?e=F0LGVc), [800d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/Eda0vUkicNpNk4oMf2eoLZkBTJbMTmymKqqZ_yoEXw98TA?e=rKu4pP)

**FastText** - [300d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/ESj0xTXmTK5Jhiocp5Oxt7IBUUmaEjczFWvQn17c2QNgcg?e=9aory9), 
[500d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/EViVRrF38fJMv1ihX2ARDNEBFFOE-MLSDHCcMG49IqEcCQ?e=g36NJ7), [800d (OneDrive)](https://witedupl-my.sharepoint.com/:u:/g/personal/dadass_wit_edu_pl/ESHkEJ7jLGlHoAIKiYdL0NkB_Z8VJyFcEHx3TpE7L1kNFg?e=FkoBgA)

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

## Language models

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
<th>Fairseq</th>
<th>Transformers</th>
</thead>
<tr>
  <td>RoBERTa&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>125k</td>
  <td>~20GB</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v3.4.0/roberta_base_transformers.zip">v3.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&#8209;v2&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>400k</td>
  <td>~20GB</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_fairseq.zip">v0.10.1</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_transformers.zip">v4.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>30k</td>
  <td>50k</td>
  <td>~135GB</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v3.4.0/roberta_large_transformers.zip">v3.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&#8209;v2&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>2k</td>
  <td>400k</td>
  <td>~200GB</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_fairseq.zip">v0.10.2</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_transformers.zip">v4.14</a>
  </td>
</tr>
  </tr>
  <tr>
  <td>DistilRoBERTa</td>
  <td>6&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>1k</td>
  <td>10ep.</td>
  <td>~20GB</td>
  <td>
  n/a
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/distilroberta_transformers.zip">v4.13</a>
  </td>
</tr>
</table>

\* L - the number of encoder blocks, H - hidden size, A - the number of attention heads <br/>

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

### BART

BART is a transformer-based sequence to sequence model trained with a denoising objective. Can be used for fine-tuning on prediction tasks, just like regular BERT, as well as various text generation tasks such as machine translation, summarization, paraphrasing etc. We provide a Polish version of BART base model, trained on a large corpus of texts extracted from Common Crawl (200+ GB). More information on the BART architecture can be found in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461). Example in HugginFace Transformers:

```python
import os
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

model_dir = "bart_base_transformers"
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BartForConditionalGeneration.from_pretrained(model_dir)
sent = "Druga<mask>światowa zakończyła się w<mask>roku kapitulacją hitlerowskich<mask>"
batch = tok(sent, return_tensors='pt')
generated_ids = model.generate(batch['input_ids'])
print(tok.batch_decode(generated_ids, skip_special_tokens=True))
# ['Druga wojna światowa zakończyła się w 1945 roku kapitulacją hitlerowskich Niemiec.']
```

Download for [Fairseq v0.10](https://github.com/sdadas/polish-nlp-resources/releases/download/bart-base/bart_base_fairseq.zip) or [HuggingFace Transformers v4.0](https://github.com/sdadas/polish-nlp-resources/releases/download/bart-base/bart_base_transformers.zip).

### GPT-2

GPT-2 is a unidirectional transformer-based language model trained with an auto-regressive objective, originally introduced in the [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) paper. The original English GPT-2 was released in four sizes differing by the number of parameters: small (112M), medium (345M), large (774M), xl (1.5B). We provide Polish versions of the medium and large GPT-2 models. Example in Fairseq:

```python
import os
from fairseq import hub_utils
from fairseq.models.transformer_lm import TransformerLanguageModel

model_dir = "gpt2_medium_fairseq"
loaded = hub_utils.from_pretrained(
    model_name_or_path=model_dir,
    checkpoint_file="model.pt",
    data_name_or_path=model_dir,
    bpe="hf_byte_bpe",
    bpe_merges=os.path.join(model_dir, "merges.txt"),
    bpe_vocab=os.path.join(model_dir, "vocab.json"),
    load_checkpoint_heads=True,
    archive_map=TransformerLanguageModel.hub_models()
)
model = hub_utils.GeneratorHubInterface(loaded["args"], loaded["task"], loaded["models"])
model.eval()
result = model.sample(
    ["Policja skontrolowała trzeźwość kierowców"],
    beam=5, sampling=True, sampling_topk=50, sampling_topp=0.95,
    temperature=0.95, max_len_a=1, max_len_b=100, no_repeat_ngram_size=3
)
print(result[0])
# Policja skontrolowała trzeźwość kierowców pojazdów. Wszystko działo się na drodze gminnej, między Radwanowem 
# a Boguchowem. - Około godziny 12.30 do naszego komisariatu zgłosił się kierowca, którego zaniepokoiło 
# zachowanie kierującego w chwili wjazdu na tą drogę. Prawdopodobnie nie miał zapiętych pasów - informuje st. asp. 
# Anna Węgrzyniak z policji w Brzezinach. Okazało się, że kierujący był pod wpływem alkoholu. [...]
```

Download [medium](https://github.com/sdadas/polish-nlp-resources/releases/download/gpt-2/gpt2_medium_fairseq.7z) or [large](https://github.com/sdadas/polish-nlp-resources/releases/download/gpt-2/gpt2_large_fairseq.7z) model for Fairseq v0.10.

### Longformer

One of the main constraints of standard Transformer architectures is the limitation on the number of input tokens. There are several known models that allow processing of long documents, one of the popular ones being Longformer, introduced in the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). We provide base and large versions of Polish Longformer model. The models were initialized with Polish RoBERTa (v2) weights and then fine-tuned on a corpus of long documents, ranging from 1024 to 4096 tokens. Example in Huggingface Transformers:

```python
from transformers import pipeline
fill_mask = pipeline('fill-mask', model='sdadas/polish-longformer-base-4096')
fill_mask('Stolica oraz największe miasto Francji to <mask>.')
```
[Base](https://huggingface.co/sdadas/polish-longformer-base-4096) and [large](https://huggingface.co/sdadas/polish-longformer-large-4096) models are available on Huggingface Hub

## Sentence encoders

### Polish transformer-based sentence encoders

The purpose of sentence encoders is to produce a fixed-length vector representation for chunks of text, such as sentences or paragraphs. These models are used in semantic search, question answering, document clustering, dataset augmentation, plagiarism detection, and other tasks which involve measuring semantic similarity between sentences. We share two models based on the [Sentence-Transformers](https://www.sbert.net/) library, trained using distillation method described in the paper [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813). A corpus of 100 million parallel Polish-English sentence pairs from the [OPUS](https://opus.nlpl.eu/) project was used to train the models. You can download them from the Hugginface Hub using the links below.

<table>
<thead>
<th>Student model</th>
<th>Teacher model</th>
<th>Download</th>
</thead>
<tr>
  <td>polish-roberta-base-v2</td>
  <td>paraphrase-distilroberta-base-v2</td>
  <td><a href="https://huggingface.co/sdadas/st-polish-paraphrase-from-distilroberta">st-polish-paraphrase-from-distilroberta</a></td>
</tr>
<tr>
  <td>polish-roberta-base-v2</td>
  <td>paraphrase-mpnet-base-v2</td>
  <td><a href="https://huggingface.co/sdadas/st-polish-paraphrase-from-mpnet">st-polish-paraphrase-from-mpnet</a></td>
</tr>
</table>

A simple example in Sentence-Transformers library:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ["Bardzo lubię jeść słodycze.", "Uwielbiam zajadać się słodkościami."]
model = SentenceTransformer("sdadas/st-polish-paraphrase-from-mpnet")
results = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
print(cos_sim(results[0], results[1]))
# tensor([[0.9794]], device='cuda:0')
```

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
- [Awesome NLP Polish (ksopyla)](https://github.com/ksopyla/awesome-nlp-polish)

### Publicly available large Polish text corpora (> 1GB)

- [OSCAR Corpus (Common Crawl extract)](https://oscar-corpus.com/)
- [CC-100 Web Crawl Data (Common Crawl extract)](http://data.statmt.org/cc-100/)
- [The Polish Parliamentary Corpus](http://clip.ipipan.waw.pl/PPC)
- [Redistributable subcorpora of the National Corpus of Polish](http://zil.ipipan.waw.pl/DistrNKJP)
- [Polish Wikipedia Dumps](https://dumps.wikimedia.org/plwiki/)
- [OPUS Parallel Corpora](https://opus.nlpl.eu/)
- [Corpus from PolEval 2018 Language Modeling Task](http://2018.poleval.pl/index.php/tasks/)

### Models supporting Polish language

#### Sentence analysis (tokenization, lemmatization, POS tagging etc.)

- [Stanza](https://stanfordnlp.github.io/stanza/) - A collection of neural NLP models for many languages from StndordNLP.
- [Trankit](https://github.com/nlp-uoregon/trankit) - A light-weight transformer-based python toolkit for multilingual natural language processing by the University of Oregon.
- [KRNNT](https://github.com/kwrobel-nlp/krnnt) and [KFTT](https://github.com/kwrobel-nlp/kftt) - Neural morphosyntactic taggers for Polish.
- [Morfeusz](http://morfeusz.sgjp.pl/) - A classic Polish morphosyntactic tagger.
- [Language Tool](https://github.com/languagetool-org/languagetool) - Java-based open source proofreading software for many languages with sentence analysis tools included.
- [Stempel](https://github.com/dzieciou/pystempel) - Algorythmic stemmer for Polish.

#### Machine translation
- [Marian-NMT](https://marian-nmt.github.io/) - An efficient C++ based implementation of neural translation models. Many pre-trained models are available, including those supporting Polish: [pl-de](https://huggingface.co/Helsinki-NLP/opus-mt-pl-de), [pl-en](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en), [pl-es](https://huggingface.co/Helsinki-NLP/opus-mt-pl-es), [pl-fr](https://huggingface.co/Helsinki-NLP/opus-mt-pl-fr), [pl-sv](https://huggingface.co/Helsinki-NLP/opus-mt-pl-sv), [de-pl](https://huggingface.co/Helsinki-NLP/opus-mt-de-pl), [es-pl](https://huggingface.co/Helsinki-NLP/opus-mt-es-pl), [fr-pl](https://huggingface.co/Helsinki-NLP/opus-mt-fr-pl).
- [M2M](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) (2021) - A single massive machine translation architecture supporting direct translation for any pair from the list of 100 languages. Details in the paper [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/pdf/2010.11125.pdf).
- [mBART-50](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) (2021) - A multilingual BART model fine-tuned for machine translation in 50 languages. Three machine translation models were published: [many-to-many](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt), [English-to-many](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt), and [many-to-English](https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt). For more information see [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401).
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) (2022) - NLLB (No Language Left Behind) is a project by Meta AI aiming to provide machine translation models for over 200 languages. A set of multilingual neural models ranging from 600M to 54.5B parameters is available for download. For more details see [No Language Left Behind: Scaling Human-Centered Machine Translation](https://research.facebook.com/publications/no-language-left-behind/).

#### Language models
- [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) (2018) - BERT (Bidirectional Encoder Representations from Transformers) is a model for generating contextual word representations. Multilingual cased model provided by Google supports 104 languages including Polish.
- [XLM-RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) (2019) - Cross lingual sentence encoder trained on 2.5 terabytes of data from CommonCrawl and Wikipedia. Supports 100 languages including Polish. See [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf) for details.
- [Slavic BERT](https://github.com/deepmipt/Slavic-BERT-NER#slavic-bert) (2019) - Multilingual BERT model supporting Bulgarian (bg), Czech (cs), Polish (pl) and Russian (ru) languages.
- [mT5](https://github.com/google-research/multilingual-t5) (2020) - Google's text-to-text transformer for 101 languages based on the T5 architecture. Details in the paper [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934).
- [HerBERT](https://huggingface.co/allegro) (2020) - Polish BERT-based language model trained by Allegro for HuggingFace Transformers in [base](https://huggingface.co/allegro/herbert-base-cased) and [large](https://huggingface.co/allegro/herbert-large-cased) variant.
- [plT5](https://huggingface.co/allegro/plt5-large) (2021) - Polish version of the T5 model available in [small](https://huggingface.co/allegro/plt5-small), [base](https://huggingface.co/allegro/plt5-base) and [large](https://huggingface.co/allegro/plt5-large) sizes.
- [XLM-RoBERTa-XL and XXL](https://github.com/pytorch/fairseq/blob/main/examples/xlmr/README.md) (2021) - Large-scale versions of XLM-RoBERTa models with 3.5 and 10.7 billion parameters respectively. For more information see [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/pdf/2105.00572.pdf).
- [mLUKE](https://huggingface.co/docs/transformers/model_doc/mluke) (2021) - A multilingual version of LUKE, Transformer-based language model enriched with entity metadata. The model supports 24 languages including Polish. For more information see [mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://arxiv.org/pdf/2110.08151.pdf).
- [XGLM](https://huggingface.co/facebook/xglm-4.5B) (2021) - A GPT style autoregressive Transformer language model trained on a large-scale multilingual corpus. The model was published in several sizes, but only the 4.5B variant includes Polish language. For more information see [Few-shot Learning with Multilingual Language Models](https://arxiv.org/abs/2112.10668).
- [PapuGaPT2](https://huggingface.co/flax-community/papuGaPT2) (2021) - Polish GPT-like autoregressive models available in [base](https://huggingface.co/flax-community/papuGaPT2) and [large](https://huggingface.co/flax-community/papuGaPT2-large) sizes.
- [mGPT](https://huggingface.co/sberbank-ai/mGPT) (2022) - Another multilingual GPT style model with 1.3B parameters, covering 60 languages. The model has been trained by Sberbank AI. For more information see [mGPT: Few-Shot Learners Go Multilingual](https://arxiv.org/abs/2204.07580). 

#### Sentence encoders
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1) (2019) - USE (Universal Sentence Encoder) generates sentence level langauge representations. Pre-trained multilingual model supports 16 langauges (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian).
- [LASER Language-Agnostic SEntence Representations](https://github.com/facebookresearch/LASER) (2019) - A multilingual sentence encoder by Facebook Research, supporting 93 languages.
- [LaBSE](https://tfhub.dev/google/LaBSE/1) (2020) - Language-agnostic BERT sentence embedding model supporting 109 languages. See [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852) for details.
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) (2020) - Sentence-level models based on the transformer architecture. The library includes multilingual models supporting Polish. More information on multilingual knowledge distillation method used by the authors can be found in [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813).
- [LASER2 and LASER3](https://github.com/facebookresearch/LASER/blob/main/nllb/README.md) (2022) - New versions of the LASER sentence encoder by Meta AI, developed as a part of the NLLB (No Language Left Behind) project. LASER2 supports the same set of languages as the first version of the encoder, which includes Polish. LASER3 adds support to less common languages, mostly low-resource African languages. See [Bitext Mining Using Distilled Sentence Representations for Low-Resource Languages](https://arxiv.org/pdf/2205.12654.pdf) for more details.

#### Optical character recognition (OCR)
- [Easy OCR](https://github.com/JaidedAI/EasyOCR) - Optical character recognition toolkit with pre-trained models for over 40 languages, including Polish.
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - Popular OCR software developed since 1980s, supporting over 100 languages. For integration with Python, wrappers such as [PyTesseract](https://github.com/madmaze/pytesseract) or [OCRMyPDF](https://github.com/ocrmypdf/OCRmyPDF) can be used. 

#### Automatic speech recognition (ASR)
- [Quartznet - Nvidia NeMo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_pl_quartznet15x5) (2021) - Nvidia NeMo is a toolkit for building conversational AI models. Apart from the framework itself, Nvidia also published many models trained using their code, which includes a speech recognition model for Polish based on Quartznet architecture.
- [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (2021) - XLS-R is a multilingual version of Wav2Vec 2.0 model by Meta AI, which is a large-scale pre-trained model for speech processing. The model is trained in a self-supervised way, so it needs to be fine-tuned for solving specific tasks such as ASR. Several fine-tuned checkpoints for Polish speech recognition exist on the HuggingFace Hub e.g. [wav2vec2-large-xlsr-53-polish](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish)
- [M-CTC-T](https://huggingface.co/speechbrain/m-ctc-t-large) (2021) - A speech recognition model from Meta AI, supporting 60 languages including Polish. For more information see [Pseudo-Labeling for Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161).
- [Whisper](https://github.com/openai/whisper/) (2022) - Whisper is a model released by OpenAI for ASR and other speech-related tasks, supporting 82 languages. The model is available in five sizes: tiny (39M params), base (74M), small (244M), medium (769M), and large (1.5B). More information can be found in the paper [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf).

#### Multimodal models
- [Multilingual CLIP (SBert)](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) (2021) - CLIP (Contrastive Language-Image Pre-Training) is a neural network introducted by [OpenAI](https://github.com/openai/CLIP) which enables joint vector representations for images and text. It can be used for building image search engines. This is a multilingual version of CLIP trained by the authors of the [Sentence-Transformers](https://www.sbert.net/) library.
- [Multilingual CLIP (M-CLIP)](https://huggingface.co/M-CLIP/M-BERT-Base-ViT-B) (2021) - This is yet another multilingual version of CLIP supporting Polish language, trained by the Swedish Institute of Computer Science (SICS).
- [LayoutXLM](https://huggingface.co/microsoft/layoutxlm-base) (2021) - A multilingual version of [LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2) model, pre-trained on 30 million  documents in 53 languages. The model combines visual, spatial, and textual modalities to solve prediction problems on visually-rich documents, such as PDFs or DOCs. See [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) and [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836) for details.
