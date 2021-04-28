# NLU Second Assignment

Student:
- **Name:** Giovanni
- **Surname:** Lorenzini
- **Student number:** 223715

## Requirements

To run this project is necessary to have `Python`, `spaCy` and `pandas`.

### Install `spaCy` with `conda`

```shell
$ conda install -c conda-forge spacy
$ python -m spacy download en_core_web_sm
```

### Install `spaCy` with `pip`

```shell
$ pip install -U pip setuptools wheel
$ pip install -U spacy
$ python -m spacy download en_core_web_sm
```

### Install `pandas` with `conda`

```shell
$ conda install pandas
```

### Install `pandas` with `pip`

```shell
$ pip install pandas
```

## Report

### 1. Evaluate `spaCy` NER on `CoNLL 2003` data

To evaluate `spaCy` NER the program first read from `conll2003/train.txt` the data and create a list of list of tuple of `(text, iob)` to be used for reference. Only the text part is also used to create a new `spaCy Doc`. The `Doc` is therefore elaborated and at the end a list of list of tuple of `(text, iob)` is obtained. To convert from `spaCy` to `CoNLL` tags I used a dictionary that I wrote myself looking at `spaCy` documentation. The two lists will be compared to obtain token-level and chuck-level perfomances.

#### 1.1. Report token-level performance (per class and total)

To obtain token-level performance the program cycle over the two lists token per token and check if `spaCy` guess and `CoNLL` reference are equal or not. The data are saved in a dictionary organized per class. At the end per class and total accuracies will be printed. The total accuracy code is inspired from `NLTK` accuracy function.

#### 1.2. Report CoNLL chunk-level performance (per class and total)

To obtain token-level performance I used the function `evaluate` from `conll.py`, providing to it the two lists that I created before. The results are then displayed using `pandas`.

### 2. Grouping of Entities

Here I have `doc.ents` were all entities are present but are not grouped and `doc.noun_chunks` were the entities are grouped but not all are present. So it's necessary to "merge" the two list to obtain a list with all and grouped entities.

#### 2.1. Write a function to group recognized named entities using `noun_chunks` method of `spaCy`

To obtain the list of grouped entities I wrote a piece of code with two indexes (`i`, `j`) that keep track of the position in the `doc.ents` list and in the `doc.noun_chunks` list. The program check if a entity in `doc.ents` is also present in `doc.noun_chunks` and if it is the case the grouped versione from `doc.noun_chunks` is added to the final list; otherwise the entity from `doc.ents` will be added.

Example:
```python
sentence = "Apple's Steve Jobs died in 2011 in Palo Alto, California."
doc = nlp(sentence)
groupedNamedEntities = groupNamedEntities(doc)
print(groupedNamedEntities)
```
Output:
```python
[['ORG', 'PERSON'], ['DATE'], ['GPE'], ['GPE']]
```

#### 2.2. Analyze the groups in terms of most frequent combinations

I applied the function that I just explained to the dataset `CoNLL 2003` to obtain a list of the most frequent combinations of NER types. I filtered out the groups with only a single NER, because I think that they are not relevant, and then used a dictionary to store the occurencies. The groups are made using `frozenset`, so the order of NER types doesn't matter. In the end the function prints a list of the ten most frequent groups.

### 3. Fix segmentation errors

The last function, `expandEntitySpan` is used to tag with `ent_type_` the missing entities that have a `compound` dependency relation. It cycles over all tokens to find the ones with a `compound` relation and if it is the case the `ent_type_` and the `ent_iob_` will be updated.

Example:
```python
sentence = "Apple's Steve Jobs died in 2011 in Palo Alto, California."
doc = nlp(sentence)
expandedEntitySpan = expandEntitySpan(doc)
print(expandedEntitySpan)
```
Output:
```python
[('Apple', 'B-ORG'), ("'s", 'O'), ('Steve', 'B-PERSON'), ('Jobs', 'I-PERSON'), ('died', 'O'), ('in', 'O'), ('2011', 'B-DATE'), ('in', 'O'), ('Palo', 'B-GPE'), ('Alto', 'I-GPE'), (',', 'O'), ('California', 'B-GPE'), ('.', 'O')]
```

## Test

Test code:
```python
train = conll.read_corpus_conll("conll2003/train.txt", fs=" ")

sentence = "Apple's Steve Jobs died in 2011 in Palo Alto, California."

nlp = spacy.load("en_core_web_sm")

refs = [[(text, iob) for text, _, _, iob in sent] for sent in train]

docs = []
hyps = []
for sent in train:
    doc = Doc(nlp.vocab, words=[w[0] for w in sent])

    for _, proc in nlp.pipeline:
        doc = proc(doc)

    hyps.append([(t.text, t.ent_iob_ +
                  ("-" if t.ent_type_ != "" else "") + spacyconll[t.ent_type_]) for t in doc])

    docs.append(doc)

print("Question 1.1:")
spacyNerTokenPerf(refs, hyps)
print("")

print("Question 1.2:")
spacyNerChunkPerf(refs, hyps)
print("")

print("Question 2.1:")
doc = nlp(sentence)
groupedNamedEntities = groupNamedEntities(doc)
print(groupedNamedEntities)
print("")

print("Question 2.2:")
namedEntitiesGroupsFrequency(docs)
print("")

print("Question 3:")
doc = nlp(sentence)
expandedEntitySpan = expandEntitySpan(doc)
print(expandedEntitySpan)
print("")
```

Output:
```
Question 1.1:
ORG     0.603896
MISC    0.646363
PER     0.844620
LOC     0.820341
total   0.826810

Question 1.2:   
              p         r         f      s
MISC   0.120760  0.539558  0.197351   3438
ORG    0.407710  0.284449  0.335104   6321
LOC    0.795632  0.699020  0.744203   7140
PER    0.770980  0.631970  0.694588   6600
total  0.407420  0.545342  0.466399  23499

Question 2.1:
[['ORG', 'PERSON'], ['DATE'], ['GPE'], ['GPE']]

Question 2.2:
({'CARDINAL', 'PERSON'}, 322)
({'NORP', 'PERSON'}, 202)
({'CARDINAL', 'ORG'}, 129)
({'PERSON', 'GPE'}, 126)
({'PERSON', 'ORG'}, 101)
({'NORP', 'CARDINAL'}, 82)
({'CARDINAL', 'GPE'}, 82)
({'ORG', 'GPE'}, 71)
({'GPE'}, 45)
({'TIME', 'DATE'}, 44)

Question 3:
[[('Apple', 'B-ORG'), ("'s", 'O'), ('Steve', 'B-PERSON'), ('Jobs', 'I-PERSON'), ('died', 'O'), ('in', 'O'), ('2011', 'B-DATE'), ('in', 'O'), ('Palo', 'B-GPE'), ('Alto', 'I-GPE'), (',', 'O'), ('California', 'B-GPE'), ('.', 'O')]]
```
