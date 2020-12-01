# -*- coding: utf-8 -*-
"""
Created on 08 Sep 2020 17:35:19
@author: jiahuei

https://universaldependencies.org/u/pos/
"""
import stanza

# English does not require Multi Word Expansion
pipeline_kwargs = dict(
    lang="en", dir="/tmp/stanza_resources", processors="tokenize,pos"
)
try:
    nlp = stanza.Pipeline(**pipeline_kwargs)
except Exception:
    stanza.download("en", dir="/tmp/stanza_resources")
    nlp = stanza.Pipeline(**pipeline_kwargs)

doc = nlp('This is tokenization done my way!\nSentence split, too!')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')

nlp = stanza.Pipeline(**pipeline_kwargs, tokenize_pretokenized=True)
doc = nlp('This is token.ization done my way!\nSentence split, too!')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')

"""
word: This	upos: PRON	xpos: DT	feats: Number=Sing|PronType=Dem
word: is	upos: AUX	xpos: VBZ	feats: Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
word: token.ization	upos: NOUN	xpos: NN	feats: Number=Sing
word: done	upos: VERB	xpos: VBN	feats: Tense=Past|VerbForm=Part
word: my	upos: PRON	xpos: PRP$	feats: Number=Sing|Person=1|Poss=Yes|PronType=Prs
word: way!	upos: NOUN	xpos: NN	feats: Number=Sing
word: Sentence	upos: NOUN	xpos: NN	feats: Number=Sing
word: split,	upos: NOUN	xpos: NN	feats: Number=Sing
word: too!	upos: PUNCT	xpos: .	feats: _
"""

doc = nlp("a girl is riding on a red bike")
print(doc.sentences)
print(doc.sentences[0].words)
print([f'{word.upos}' for sent in doc.sentences for word in sent.words])
doc = nlp("a girl is riding on a red bike")
print([f'{word.upos}' for sent in doc.sentences for word in sent.words])
doc = nlp("a girl is riding on a red bike")
print(" ".join(word.upos for sent in doc.sentences for word in sent.words))

