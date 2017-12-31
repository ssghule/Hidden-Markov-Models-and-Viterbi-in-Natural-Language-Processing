# POS Tagging using Hidden Markov Models

Natural language processing (NLP) is an important research area in artificial intelligence, dating back to
at least the 1950’s. One of the most basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step
towards extracting semantics from natural language text.</br></br>

__Data:__ The dataset is a large corpus of labeled training and testing data,
consisting of nearly 1 million words and 50,000 sentences. The file format of the datasets is:
each line consists of a word, followed by a space, followed by one of 12 part-of-speech tags: ADJ (adjective),
ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON
(pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). Sentence boundaries are
indicated by blank lines. </br></br>

label.py is the main program, pos scorer.py, which has the scoring code, and pos solver.py, which contains the actual
part-of-speech estimation code. The program takes as input two filenames: a training file and a testing file and displays accuracy using simple probability, Bayes net variable elimination method and Viterbi algorithm to find the maximum a posteriori (MAP). </br> </br>
It also displays the logarithm of the posterior probability for each solution it finds, as well as a
running evaluation showing the percentage of words and whole sentences that have been labeled correctly
according to the ground truth. </br></br>
To run the code:</br>
__python label.py part2 training_file testing_file__

