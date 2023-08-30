# Questions: An AI can answer to your questions

Harvard CS50AI Project

## Description:

An AI program that answers questions, by determining the most relevant document(s) using tf-idf ranking and then extracting the most relevant sentence(s) using idf and a query term density measure.

## Tech Stack:

* Python
* NLTK

## Background:

Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. Among the more famous question answering systems is Watson, the IBM computer that competed (and won) on Jeopardy!. A question answering system of Watson’s accuracy requires enormous complexity and vast amounts of data, but in this problem, we’ll design a very simple question answering system based on inverse document frequency.

Our question answering system will perform two tasks: document retrieval and passage retrieval. Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

How do we find the most relevant documents and passages? To find the most relevant documents, we’ll use tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once we’ve found the most relevant documents, there many possible metrics for scoring passages, but we’ll use a combination of inverse document frequency and a query term density measure (described in the Specification).

More sophisticated question answering systems might employ other strategies (analyzing the type of question word used, looking for synonyms of query words, lemmatizing to handle different forms of the same word, etc.) but we’ll leave those sorts of improvements as exercises for you to work on if you’d like to after you’ve completed this project!

## Project Specification:

### load_files
The load_files function should accept the name of a directory and return a dictionary mapping the filename of each .txt file inside that directory to the file’s contents as a string.

### tokenize
The tokenize function should accept a document (a string) as input, and return a list of all of the words in that document, in order and lowercased.

### compute_idfs
The compute_idfs function should accept a dictionary of documents and return a new dictionary mapping words to their IDF (inverse document frequency) values.

### top_files
The top_files function should, given a query (a set of words), files (a dictionary mapping names of files to a list of their words), and idfs (a dictionary mapping words to their IDF values), return a list of the filenames of the the n top files that match the query, ranked according to tf-idf.

### top_sentences
The top_sentences function should, given a query (a set of words), sentences (a dictionary mapping sentences to a list of their words), and idfs (a dictionary mapping words to their IDF values), return a list of the n top sentences that match the query, ranked according to IDF.

## How to run

1. Clone this project
2. Install requirements package:
   ```
   pip install -r requirements.txt
   ```
3. Run the AI Model:
   ```
   python questions.py corpus
   ```
   then type your questions, AI Model will answer it.
