import nltk
import sys
import os
import string
import math

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
    # Create a dictionary to map the filename of each text file
    d = {}

    # Loop through each file in the directory
    for text_file in os.listdir(directory):

        # Check if it is a text file
        if text_file.endswith(".txt"):

            # Read the file
            with open(os.path.join(directory, text_file), encoding="utf-8") as f:

                # Get the string from text file then append to the dictionary
                d[text_file] = f.read()

    # Return the dictionary
    return d


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Return a list of words from sentence
    return list(
        word for word in nltk.word_tokenize(document.lower())
        if word not in nltk.corpus.stopwords.words("english") and word not in string.punctuation and any(c.isalpha() for c in word)
    )


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Create a dictionary to store all IDF values
    idfs = {}

    # Loop through each document in documents dictionary
    for document in documents:

        # Loop through each word in words set
        for word in set(documents[document]):

            # Calculate the IDF value with smoothing then append to dictionary
            idfs[word] = (math.log(len(documents) / sum(word in documents[document] for document in documents))
                          if sum(word in documents[document] for document in documents) > 0 else math.log(len(documents) + 2))

    # Return the dictionary
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Create a dictionary to store all the tf-idf values
    tfidfs = {file: 0 for file in files}

    # Loop through each file in files dictionary
    for file in files:

        # Loop through each word in query set
        for word in query:

            # Check if word is in the file
            if word in files[file]:

                # Get the tf value
                tf = files[file].count(word)

                # Calculate the tf-idf value then add to the dictionary
                tfidfs[file] += (tf * idfs[word])

    # Return sorted list of top n files
    return sorted(tfidfs, key=lambda x: tfidfs[x], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create a dictionary to store  “matching word measure” for each sentence
    d = {}

    # Loop through each sentence in sentences dictionary
    for sentence in sentences:

        # Create a variable score to store “matching word measure”
        score = 0

        # Create a variable to count word appear in both query and sentence
        counter = 0

        # Loop through each word in the query set
        for word in query:

            # Check if the word in sentence
            if word in sentences[sentence]:

                # Update score with idf value of word
                score += idfs[word]

                # Update count
                counter += sentences[sentence].count(word)

        # Calculate query term density for sentence
        density = counter / len(sentences[sentence])

        # Add to dictionary
        d[sentence] = (score, density)

    # Return sorted list of the `n` top sentences that match the query in order idf value, query term density
    return sorted(d, key=lambda x: (d[x][0], d[x][1]), reverse=True)[:n]


if __name__ == "__main__":
    main()
