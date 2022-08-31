import string

import nltk
from lxml import etree
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Read .xml string
xml_string = './news.xml'
tree = etree.parse(xml_string)
root = tree.getroot()

# An empty dataset, used in RF-IDF, where all lemmatizated nouns are stored
dataset = []


# Parent tag. Look through child tags in a root tag
for j in range(len(root[0])):
    # Child tag. Look through child tags in a parent tag
    for i in root[0][j]:
        if i.get('name') == 'text':
            # text tokenization
            tokenized_text = word_tokenize(i.text.lower())

            # Lemmatization of words
            wordnet = WordNetLemmatizer()
            lemma_tokenized_text = [wordnet.lemmatize(word, pos='n') for word in tokenized_text]

            # clear all punctuation marks
            lemma_tokenized_text_pure = [word for word in lemma_tokenized_text if word not in string.punctuation
                                         and word not in stopwords.words('english')]

            # create a list of nouns
            tokenized_nouns = [nltk.pos_tag([word])[0][0] for word in lemma_tokenized_text_pure
                               if nltk.pos_tag([word])[0][-1] == 'NN']

            # form a complete dataset to use in RF-IDF
            dataset.append(' '.join(tokenized_nouns))


# use sklearn and build TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset)

# dictionary with vocabulary and position in the text
index_value = {i[1]: i[0] for i in vectorizer.vocabulary_.items()}

# create a list of dictionaries for each text
fully_indexed = []
for row in tfidf_matrix:
    fully_indexed.append({index_value[column]: value for (column, value) in zip(row.indices, row.data)})

# Iterate through head name, print it and then print five the most frequent words
for n, j in enumerate(range(len(root[0]))):
    # Child tag. Look through child tags in a parent tag
    for i in root[0][j]:
        if i.get('name') == 'head':
            # Print the head
            print(f'{i.text}:')

            # choose five the most frequent words
            freq_words = sorted(fully_indexed[n].items(), key=lambda x: (x[1], x[0]), reverse=True)[:5]
            words = [word[0] for word in freq_words]

            # Print the words
            print(' '.join(words))
