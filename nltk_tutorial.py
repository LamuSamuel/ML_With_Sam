# what is Nltk ?
# one of the largest python libraries for performing various Natural Language Processing tasks
# Some important NLTK important tasks : Tokenization, Stemming, Lemmatization, and POS Tagging

# installation
# pip/pip3 install nltk

# to download all the nltk packages we can use nltk.download('all')

# to download a specific package nltk.download('package_name')
'''
Tokenization  - breaking down the text into smaller units , para to sentences , sentences to words.
this will  be the initial step in any nlp pipeline.
lets look into an example :
before that sent_tokenize will split a para or block of text into individual sentence and
word_tokenize will split a sentence into individual words
'''



import nltk
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt_tag')

smoke_tokenize = "Samuel is learning about NLTK! and he is currently testing how to break a sentence"
first_break=sent_tokenize(smoke_tokenize)
for sentence in first_break:
    print(word_tokenize(sentence))

# The above way (sent_tokenize) breaks a text block in to 'n' times when ever there is a . , or ! , and the next part of sentence is stored in another list.
# To break everything in to a single list just use word_tokenize as below

smoke_tokenize = "Samuel is learning about NLTK! and he is currently testing how to break a sentence"
print(word_tokenize(smoke_tokenize)) , # :) simple


# What is Stemming
# This package provides the base words for the present continuous , past and future .
# It drops the affixes , Stemmers are fast and computationally less expensive than lemmatizers
from nltk import PorterStemmer , sent_tokenize , word_tokenize

smoke_stemmer = "Samuel is Playing, while Watching and breathing"
porter = PorterStemmer()
break_some_stemmer_sentence= sent_tokenize(smoke_stemmer)
for sentence in break_some_stemmer_sentence:
    tokenized_words_for_stemming = word_tokenize(sentence)
    for word in tokenized_words_for_stemming:
        print(porter.stem(word))

#The output is like this , while it is not grammatically correct our intention is to stem the words to their base.
'''
lamusamuel@Samuels-MacBook-Pro ML % python3 nltk_tutorial.py
samuel
is
play
,
while
    watch
and
breath
'''

# Lemmatization : means grouping together the inflected form of same words , while considering the word's meaning and part of speech

# Unlike stemming, which just chops off endings roughly, Lemmatization looks at the word’s meaning and part of speech to find the correct base form
# slow and computationally expensive than the stemmers

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
lemm_text = "Samuel is coding while learning and thinking"
#lemm_text = "children are playing"
break_words  = word_tokenize(lemm_text)
lemmatized_words = [lemmatizer.lemmatize(word) for word in break_words]
print("original text",lemm_text)
print("lemmatized_text",' '.join(lemmatized_words))

'''
The output remains the same because lemmatization defaults to noun forms unless which part of speech it is so all coding , leaning and thinking are treated as nouns
but in the line 74 we have another text and children is lemmatized to child because its a plural noun so WordNetLemmatizer knows if its plural or singular so even without POS it recognises.
but to make our line 73 work we need to introduce POS in our model ,
basics NN - Noun , NNS - means noun plural , VB -verb , JJ - Adjective , RB - Adverb and so on
so we will call treebank_tag this package is embedded in POS and is vey much smarted to recognize which word in the usage belongs to what POS
This treebank_tag will recognize the words POS and calls the respective POS when needed .
Lets dive into an example.
'''
from nltk import pos_tag
from nltk.corpus import  wordnet # corpus means dump of all natural language data sets
nltk.download('averaged_perceptron_tagger_eng') # for taagin the parts of speech

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer()
lemm_text_enhanced = "Samuel is coding while learning and thinking"
tokenise_lem_enhanced = word_tokenize(lemm_text_enhanced)
tagged = pos_tag(tokenise_lem_enhanced)
# We print the tagged it looks like this : [('Samuel', 'NNP'), ('is', 'VBZ'), ('coding', 'VBG'), ('while', 'IN'), ('learning', 'VBG'), ('and', 'CC'), ('thinking', 'VBG')]

#so we have two values words and their respected pos, lets now use  `the get_wordnet_pos(pos))` function to get the base POS for the words like coding , thinking and learning
lemmatized_words_enhanced = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged]
print("original text:",lemm_text_enhanced)
print("Lemmatized sentences",' '.join(lemmatized_words_enhanced))

## output
# original text: Samuel is coding while learning and thinking
# Lemmatized sentences Samuel be cod while learn and think
# even though it's not grammatically correct we were able to get the concept of lemmatization.

# both stemming and lemmitization looks same for our example sentence, but look below how they differ for other texts.

##                 Word | Stemmer Output | Lemmatizer Output |

##                 studies | studi | study |

##                 better | better | good |

##                 university | univers |  university |

##                  was |        wa |            be |

