import nltk
import sys
import os
import string
import math
import operator

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    print(filenames)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files=dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename),encoding="utf-8") as f:
            files[filename]=f.read()
    return files

def filterfun(word):
    if(word in string.punctuation or word in nltk.corpus.stopwords.words("english")):
        return False
    else:
        return True


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.
 
    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    contents=[
        word.lower() for word in
        nltk.word_tokenize(document)
        if word.isalpha()
    ]
    filtered=list(filter(filterfun,contents))
    return filtered


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words=set()
    for filename in documents:
        words.update(documents[filename])
    idfs=dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs=[]

    for filename in files:
        tf_idf=0
        for word in query:
            if(word in files[filename]):
                tf_idf+=list(files[filename]).count(word)*idfs[word]
        tfidfs.append((filename,tf_idf))
    
    sorted_tfidfs = sorted(tfidfs,key=lambda sl: -sl[1])
    top_n_files=[]
    for i in range(0,n):
        top_n_files.append(sorted_tfidfs[i][0])
    
    return top_n_files

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    sentence_idfs=[]

    for sentence in sentences:
        matched_words=0
        idf=0
        for word in query:
            if(word in sentences[sentence]):
                matched_words+=1
                idf+=idfs[word]
        query_term_density=matched_words/len(sentences[sentence])
        sentence_idfs.append((sentence,idf,query_term_density))
    
    sorted_sentence_idfs=sorted(sentence_idfs,key=lambda sl: (-sl[1],-sl[2]))
    top_n_sentences=[]
    for i in range(0,n):
        top_n_sentences.append(sorted_sentence_idfs[i][0])
    
    return top_n_sentences


if __name__ == "__main__":
    main()
