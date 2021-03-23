from io import open

from nltk import FreqDist
from nltk import WittenBellProbDist
from nltk.util import bigrams
from conllu import parse_incr
import numpy as np
import pandas as pd

########## Getting data from corpora ##########

# CHOOSE LANGUAGE from corpora below
lang = 'en'

corpora = {}
corpora['en'] = 'UD_English-EWT/en_ewt'
corpora['es'] = 'UD_Spanish-GSD/es_gsd'
corpora['nl'] = 'UD_Dutch-Alpino/nl_alpino'
corpora['cz'] = 'UD_Czech-CAC/cs_cac'
corpora['ge'] = 'UD_German-GSD/de_gsd'
corpora['pe'] = 'UD_Persian-PerDT/fa_perdt'

def train_corpus(lang):
	return corpora[lang] + '-ud-train.conllu'

def test_corpus(lang):
	return corpora[lang] + '-ud-test.conllu'

# Remove contractions such as "isn't".
def prune_sentence(sent):
	return [token for token in sent if type(token['id']) is int]

def conllu_corpus(path):
	data_file = open(path, 'r', encoding='utf-8')
	sents = list(parse_incr(data_file))
	return [prune_sentence(sent) for sent in sents]

# Limit length of sentences to avoid underflow.
max_len = 100

# Get sentences for training and testing 
train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
test_sents = [sent for sent in test_sents if len(sent) <= max_len]
print(len(train_sents), 'training sentences')
print(len(test_sents), 'test sentences')

# For language comparison - uncomment if you want to run it and comment out: Get sentences for training and testing 
# test_leng = 400
# train_leng = 12000
# train_sents_corp = conllu_corpus(train_corpus(lang))
# train_sents_corp = [sent for sent in train_sents_corp if len(sent) <= max_len]
# test_sents_corp = conllu_corpus(test_corpus(lang))
# test_sents_corp = [sent for sent in test_sents_corp if len(sent) <= max_len]
# test_sents=[]
# for i in range(test_leng):
#     test_sents.append(test_sents_corp[i])
# train_sents=[]
# for i in range(train_leng):
#     train_sents.append(train_sents_corp[i])

print(len(train_sents), 'training sentences')
print(len(test_sents), 'test sentences')

########## FUNCTIONS ##########

# TRAINING DATA in tuple
# Gives a tuple of words and tags
def get_words_and_tags(sentences):
    words = []
    tags = []
    for sent in sentences:
	    for token in sent:
		    words.append(token['form'])
		    tags.append(token['upos'])
    # put it in the list of tuples
    list_of_tup = list(zip(words,tags))
    return list_of_tup

# Returns lemmas from a given corpora
def get_lemmas(sentences):
    lemmas=[]
    for sent in sentences:
        for token in sent:
            lemmas.append(token['lemma'])
    return lemmas

# BIGRAMS of training tags
# This function creates bigrams from our training set and adds tags for begining and end of the sentence
def bigrams_tags_begend(train_sents):
    start = '<s>'
    end = '</s>'
    tags = []
    for sent in train_sents:
        tags.append(start)
        for token in sent:
            tags.append(token['upos'])
        tags.append(end)
    tags_bigrams = list(bigrams(tags))
    return tags_bigrams

# EMISSION
# Counts for Emission probability
# Returns: Emission probability. Calculated by count of the word marked by the given tag and count of that tag
def emission_by_count(train_words_tags, given_word, given_tag):
    word_selection = [word for (word, tag) in train_words_tags if word == given_word and tag==given_tag]
    word_count = len(word_selection)
    tag_selection = [tag for (word, tag) in train_words_tags if tag==given_tag]
    tag_count = len(tag_selection)
    return (word_count/tag_count)
# SMOOTH EMISSION
def emission_by_smoothing(train_words_tags):
    smoothed = {}
    uniqueTags = set([t for (_,t) in train_words_tags])
    for tag in uniqueTags:
        words = [w for (w,t) in train_words_tags if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
    return smoothed
# To calculate emission probability for particular word and tag in test set
def emission_smoothed_proba(smoothed, given_word, given_tag):
    return smoothed[given_tag].prob(given_word)

# TRANSITION
def transition_by_count(train_tags_bigrams, tag1, tag2):
    tag1_tag2 = [pair for pair in train_tags_bigrams if pair[0]==tag1 and pair[1]==tag2]
    tag1_tag2_count = len(tag1_tag2)
    tag1_occurance = [t1 for t1,t2 in train_tags_bigrams if t1 == tag1]
    tag1_count = len(tag1_occurance)
    return (tag1_tag2_count/tag1_count)
# SMOOTH TRANSITION
# Where tag1 is preceeding, tag2 is current tag
def transition_by_smoothing(train_tags_bigrams):
    smoothed = {}
    uniqueTags = set([t for (t,_) in train_tags_bigrams])
    for tag1 in uniqueTags:
        tag2 = [t2 for (t1,t2) in train_tags_bigrams if t1 == tag1]
        smoothed[tag1] = WittenBellProbDist(FreqDist(tag2), bins=1e5)
    return smoothed
# To calculate transition probability for particular word and tag in test set
def transition_smoothed_proba(smoothed, given_tag1, given_tag2):
    return smoothed[given_tag1].prob(given_tag2)

# Greedy algorithm
def Greedy(test_sents, train_words_tags, train_tags_bigrams):
    smoothed_transition = transition_by_smoothing(train_tags_bigrams)
    smoothed_emission = emission_by_smoothing(train_words_tags)
    predicted_tags=[]
    words_to_tag = []
    tags = list(set([t for w,t in train_words_tags]))
    for sent in test_sents:
        words = []
        for token in sent:
            words.append(token['form'])
            words_to_tag.append(token['form'])
        for key, word in enumerate(words):
            tag_probabilities = []
            for tag in tags:
                # transition probability
                if key==0:
                    transition_probability = transition_smoothed_proba(smoothed_transition,'<s>', tag)
                else:
                    transition_probability = transition_smoothed_proba(smoothed_transition, predicted_tags[-1], tag)
                # emission probability
                emission_probability = emission_smoothed_proba(smoothed_emission, word, tag)
                tag_probability = emission_probability*transition_probability
                tag_probabilities.append(tag_probability)
            tag_proba_max = max(tag_probabilities)
            tag_max = tags[tag_probabilities.index(tag_proba_max)]
            predicted_tags.append(tag_max)
    return list(zip(words_to_tag, predicted_tags))

# Prints transition matrix
def print_transition_matrix(train_tags_bigrams, tags):
    smoothed = transition_by_smoothing(train_tags_bigrams)
    tags.insert(0,'<s>')
    tags.append('</s>')
    transition_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for tag1 in tags:
        for tag2 in tags:
            transition_matrix[tags.index(tag1), tags.index(tag2)] = transition_smoothed_proba(smoothed, tag1, tag2)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    tags_df = pd.DataFrame(transition_matrix, columns = list(tags), index=list(tags))
    print(tags_df)

# VITERBI algorithm
def Viterbi(test_sents, train_words_tags, train_tags_bigrams):
    smoothed_transition = transition_by_smoothing(train_tags_bigrams)
    smoothed_emission = emission_by_smoothing(train_words_tags)
    predicted_tags=[]
    words_to_tag = []
    tags = list(set([t for w,t in train_words_tags]))
    print(f'Unique tags: {tags}')

    # Take training sentences and put them into words, the algorithm takes sentences and run algorithm on them one by one
    for sent in test_sents:
        words = []
        for token in sent:
            words.append(token['form'])
            words_to_tag.append(token['form'])

        # Creates viterbi matrix that contains cells that have 2 values in them, first one is probability of that cell and second one is index of previous tag
        cols = len(words)
        rows = len(tags)
        viterbi_matrix = np.zeros((rows,cols,2))

        # Initialise first column of the matrix
        for current_tag_index in range(0, len(tags)):
            viterbi_matrix[current_tag_index, 0, 0] =  transition_smoothed_proba(smoothed_transition, '<s>',tags[current_tag_index])*emission_smoothed_proba(smoothed_emission, words[0],
             tags[current_tag_index])
            viterbi_matrix[current_tag_index, 0, 1] = 0 #here it does not matter what number it has because previous tag is <s> start of the sentence and it will never be considered
        for word_index in range(1, len(words)):
            for current_tag_index in range(len(tags)):
                cell_proba = [] # represents cell and inside goes all the probabilities gained from previous column that have same order as tags => its index will give us index of te tag later
                for previous_tag_index in range(len(tags)):
                    cell_proba.append(viterbi_matrix[previous_tag_index, word_index-1, 0]*transition_smoothed_proba(smoothed_transition, tags[previous_tag_index],tags[current_tag_index]))
                max_cell_proba = max(cell_proba)
                max_cell_tag_index = cell_proba.index(max_cell_proba) # as mentioned columns in the matrix are indexed in the same way the tags
                viterbi_matrix[current_tag_index, word_index, 0] = max_cell_proba*emission_smoothed_proba(smoothed_emission, words[word_index], tags[current_tag_index])
                viterbi_matrix[current_tag_index, word_index, 1] = max_cell_tag_index
        last_proba = []
        # Final calculation of probability and terminating Veterbi algorithm
        for previous_tag_index in range(len(tags)):
            last_proba.append(viterbi_matrix[previous_tag_index, len(words)-1,0]*transition_smoothed_proba(smoothed_transition, tags[previous_tag_index],'</s>'))
        max_last_proba = max(last_proba)
        max_last_tag_index = last_proba.index(max_last_proba)

        # Backtracking
        backtrack = []
        predicted_tag_index =max_last_tag_index
        backtrack.append(tags[predicted_tag_index])
        for column in range(len(words)-1,0,-1): # This goes back column by column
            predicted_tag_index = int(viterbi_matrix[predicted_tag_index, column, 1])
            backtrack.append(tags[predicted_tag_index])
        backtrack.reverse()
        for t in backtrack:
            predicted_tags.append(t)

    print("Predicted words:", len(words_to_tag))
    print("Predicted tags: ", len(predicted_tags))
    return list(zip(words_to_tag, predicted_tags))

########## Main part ##########

# TUPLE of training data - words and tags in tuple
train_words_tags = get_words_and_tags(train_sents)

### RUNNING Viterbi and greedy algorithm
test_words_tags = get_words_and_tags(test_sents)
test_correct_tags = [t for w,t in test_words_tags]
test_words = [w for w,t in test_words_tags]
train_tags_bigrams = bigrams_tags_begend(train_sents)
predicted_words_tags = Viterbi(test_sents, train_words_tags, train_tags_bigrams)
predicted_tags = [t for w,t in predicted_words_tags]
predicted_words_tags_greedy = Greedy(test_sents, train_words_tags, train_tags_bigrams)
predicted_tags_greedy = [t for w,t in predicted_words_tags_greedy]
# ACCURACY
check = [i for i, j in zip(predicted_tags, test_correct_tags) if i == j]
accuracy = len(check)/len(test_correct_tags)
print(f'Viterbi Algorithm Accuracy: {accuracy*100} %')
check_greedy = [i for i, j in zip(predicted_tags_greedy, test_correct_tags) if i == j]
accuracy_greedy = len(check_greedy)/len(test_correct_tags)
print(f'Greedy Algorithm Accuracy: {accuracy_greedy*100} %')

######## INSPECTION of data ########
tags = [tag for (word, tag) in train_words_tags]
words = [word for (word, tag) in train_words_tags]
lemmas = get_lemmas(train_sents)
print("Number of words in training set: ", len(words))
print("Number of unique words in training set: ", len(set(words)))
print("Number of unique tags in training set: ", len(set(tags)))
print("Number of unique lemmas in training set: ", len(set(lemmas)))
print(f"Forms/Lemmas ratio: {len(set(words))/len(set(lemmas))}")
print_transition_matrix(train_tags_bigrams, list(set(tags)))

#Find and output the most common tag
wordsFreq = FreqDist(words)
tagsFreq = FreqDist(tags)
#wordsFreq = FreqDist(words)
#posFreq = FreqDist(tags)
#print(posFreq.most_common)
tagsFreq.plot(title = f'Frequency Distribution of Tags - {lang}')

#TESTING smoothing vs no smoothing for english corpora
# print("Transition no smoothing: ", transition_by_count(train_tags_bigrams, 'NOUN', 'ADP'))
# print("Transition smoothing: ", transition_smoothed_proba(transition_by_smoothing(train_tags_bigrams), 'NOUN', 'ADP'))
# print("Emission no smoothing: ", emission_by_count(train_words_tags, "dog", "NOUN"))
# print("Emission smoothing: ", emission_smoothed_proba(emission_by_smoothing(train_words_tags), "dog", "NOUN"))
