# Create a concept list from captions
# Jiaxin Wu
# 2020.02.04

from __future__ import print_function
from collections import Counter
import json
import argparse
import os
import sys
import re
from textblob import TextBlob
from util.util import makedirsforfile, checkToSkip,Progbar
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.parse import CoreNLPParser
from util.constant import ROOT_PATH
wnl = WordNetLemmatizer()


pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower()

def from_txt(txt,spliter=' '):
    captions = []
    cap_ids = []
    with open(txt, 'r', encoding='iso-8859-1') as reader:
        for line in reader:
            if spliter=='::':
                cap_id = line.split('::')[0]
                caption = line.split('::')[1]
            else:
                cap_id= line.split(' ')[0]
                caption = ' '.join(line.split(' ')[1:])


            cap_ids.append(cap_id)
            captions.append(caption.strip())
    return cap_ids,captions

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_with_postag(sentence,verb_only=False):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = []
    for wd, tag in words_and_tags:
        lemma_word = wd
        if verb_only:
            if tag == 'v':
                lemma_word = wd.lemmatize(tag)
        lemmatized_list.append(lemma_word)
    return lemmatized_list

def get_lemma(sent,verb_only=False):
    lemmas = []
    tagged_sent =  list(pos_tagger.tag(sent.split()))

    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemma_word = tag[0]
        if verb_only:
            if wordnet_pos=='v':
                lemma_word = wnl.lemmatize(tag[0], pos=wordnet_pos)

        lemmas.append(lemma_word)
    return lemmas

def build_concept(collection,rootpath=ROOT_PATH):
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    cap_file = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt'%collection)
    cap_ids,captions = from_txt(cap_file)


    pbar = Progbar(len(captions))
    capId_captions_words = []

    word_gt_file = os.path.join(rootpath, collection, 'TextData',
                                collection + '.caption.txt.with_lemma_concept_word')
    writer = open(word_gt_file, 'w')

    stop_word_file = os.path.join(ROOT_PATH, 'stopwords_en.txt')
    stop_words = []
    with open(stop_word_file, 'rb') as reader:
        for word in reader:
            word = word.decode().strip()
            stop_words.append(word)

    for i, caption in enumerate(captions):
        cap_id = cap_ids[i]
        caption = clean_str(caption).strip()
        tokens = get_lemma(caption,verb_only=True)
        line = '%s::%s::%s'%(cap_id,caption,','.join(tokens))
        capId_captions_words.append(line)
        writer.write(line+'\n')

        counter.update(tokens)

        pbar.add(1)
        # if i % 1000 == 0:
        #     print("[%d/%d] tokenized the captions." % (i, len(captions)))

    return capId_captions_words,counter



def main(option):
    rootpath = option.rootpath
    collection = option.collection
    threshold = option.threshold

    counter_file = os.path.join(rootpath, collection, 'TextData', 'concept',
                                'concept_frequency_count_gt%s.concept_word.txt' % threshold.split(',')[0])

    if checkToSkip(counter_file, option.overwrite):
        sys.exit(0)
    makedirsforfile(counter_file)

    stop_word_file = os.path.join(ROOT_PATH, 'stopwords_en.txt')
    stop_words = []
    with open(stop_word_file, 'rb') as reader:
        for word in reader:
            word = word.decode().strip()
            stop_words.append(word)

    capId_captions_words,concept_counter = build_concept(collection, rootpath=rootpath)


    for ithreshold in threshold.split(','):
        ithreshold = int(ithreshold)
        counter_file = os.path.join(rootpath, collection, 'TextData', 'concept',
                                    'concept_frequency_count_gt%s.concept_word.txt' % threshold.split(',')[0])

        concept_counter_list = []
        for word, cnt in concept_counter.items():
            if cnt >= ithreshold:
                if not word in stop_words:
                    concept_counter_list.append([word, cnt])
        concept_counter_list.sort(key=lambda x: x[1], reverse=True)
        with open(counter_file, 'w') as writer:
            writer.write('\n'.join(map(lambda x: x[0] + ' %d' % x[1], concept_counter_list)))
        print("Saved vocabulary counter file to %s", counter_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='root path. (default: %s)'%ROOT_PATH)
    parser.add_argument('collection', type=str, help='collection tgif|msrvtt10k')
    parser.add_argument('--threshold', type=str, default='5', help='threshold to build vocabulary. (default: 5)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed vocabulary file. (default: 0)')

    opt = parser.parse_args()
    print(json.dumps(vars(opt), indent = 2))

    main(opt)

