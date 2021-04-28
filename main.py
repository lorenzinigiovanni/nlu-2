import spacy
from spacy.tokens import Doc
import conll
import pandas
from spacyconll import spacyconll


def spacyNerTokenPerf(refs, hyps):
    l = []
    for i in range(len(refs)):
        for j in range(len(refs[i])):
            l.append((refs[i][j][-1], hyps[i][j][-1]))

    accuracy = sum(x == y for x, y in l) / len(l)

    classes = {}

    for x in l:
        ref = conll.parse_iob(x[0])
        hyp = conll.parse_iob(x[1])

        if ref[1] != None:
            if hyp[1] != None:
                if ref[1] in classes:
                    classes[ref[1]][0] += 1
                else:
                    classes[ref[1]] = [1, 0]

                if ref == hyp:
                    classes[ref[1]][1] += 1

    for key, value in classes.items():
        print("{}\t{:.6f}".format(key, value[1] / value[0]))
    print("total\t{:.6f}".format(accuracy))


def spacyNerChunkPerf(refs, hyps):
    results = conll.evaluate(refs, hyps)

    pd_tbl = pandas.DataFrame().from_dict(results, orient="index")
    pd_tbl.round(decimals=3)

    print(pd_tbl)


def groupNamedEntities(doc):
    list1 = []

    ents = doc.ents
    chunks = list(doc.noun_chunks)

    i = 0
    j = 0
    while i < len(ents):
        list2 = []

        while j < len(chunks) and len(chunks[j].ents) == 0:
            j += 1

        if j < len(chunks) and len(chunks[j].ents) != 0 and ents[i] == chunks[j].ents[0]:
            for ent in chunks[j].ents:
                list2.append(ent.label_)
                i += 1
            j += 1
        else:
            list2.append(ents[i].label_)
            i += 1

        list1.append(list2)

    return list1


def namedEntitiesGroupsFrequency(docs):
    freq = {}

    for doc in docs:
        groupedNamedEntities = groupNamedEntities(doc)
        for group in groupedNamedEntities:
            if len(group) > 1:
                s = frozenset(group)
                if (s in freq):
                    freq[s] += 1
                else:
                    freq[s] = 1

    for key, value in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:10]:
        print((set(key), value))


def expandEntitySpan(doc):
    ent_iobs = [t.ent_iob_ for t in doc]
    ent_types = [t.ent_type_ for t in doc]

    for token in doc:
        if token.head.ent_type_ != "" and token.dep_ == "compound":
            ent_types[token.i] = token.head.ent_type_

            if token.i < token.head.i and token.head.i == "B":
                ent_iobs[token.i] = "B"
                ent_iobs[token.head.i] = "I"
            elif token.i > token.head.i:
                ent_iobs[token.i] = "I"
            else:
                ent_iobs[token.i] = "B"

    return [(t.text, ent_iob + ("-" if ent_type != "" else "") + ent_type) for t, ent_iob, ent_type in zip(doc, ent_iobs, ent_types)]


train = conll.read_corpus_conll("conll2003/train.txt", fs=" ")[:100]

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
