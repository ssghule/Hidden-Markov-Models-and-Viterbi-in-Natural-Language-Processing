#!/usr/bin/env python
###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
# Saurabh Agrawal  -  agrasaur
# Gaurav Derasaria -  gderasar
# Sharad Ghule     -  ssghule
# (Based on skeleton code by D. Crandall)
#
#
####
# How we are representing this problem?
# A) We have mapped this problem into a Hidden Markov model. We consider each test word in the test sentence as a
# new observed state. And we try to compute the hidden states (most likely part of speech tag for the observed state).
#
# Hidden Markov model requires calculation of three probability tables:
# i) Initial Probabilities: We compute these probabilities from the training text file. These are the probabilities
#  of a part of speech being the first word of a statement. For computing initial probabilities, we traversed the
#  training file and counted the first words of the sentence, and then normalized these probabilities.
#
# ii) Transition probabilities: This probability table stores the probability of transitioning from one part of
#  speech to another. For computing this probability table, we counted the occurrences of a particular part of speech
#  after another part of speech.
#
# iii) Emission probabilities: These are the probabilities of the a word having a particular part of speech. For
#  computing these probabilities, we traversed all the words and counted the occurrences of different parts of
#  speeches for a word. We normalized these probabilities and divided it by the overall tag probabilities to
#  determined emission probabilities.
#
# How we did part of speech tagging?
# i) Simplified: The algorithm took into account only the emission and POS tag probabilities. There was no connection
#  between the two hidden states. The part of speech assigned to each word in a sentence was simply decided by the
#  emission probability and the probability of the POS tag occurring.
#
# ii) Variable Elimination: The algorithm uses all three probabilities. At every transition we compute the sum of all
# the probabilities from the previous POS tag times the transisition from previous POS tag to the current word. Then
# we would multiply the sum with the emission probability for that word. Finally, we assign the computed value to the
# word. The train POS tag with the highest probability would be the assigned POS tag.
#
# iii) Viterbi Algorithm: Again, the algorithm considers all three probabilities. The Viterbi algorithm stores the
#  most likely previous POS tag and the probability of transitiong from the previous POS tag to the current POS tag
#  times the emission probability of the current word. When the complete table was populated we backtrack from the
#  most likely state at the end always going to the previous state at the index stored on the current character.
#
# Output:
# ==> So far scored 2000 sentences with 29442 words.
# Words correct:     Sentences correct:
# 0. Ground truth:      100.00%              100.00%
#                       1. Simplified:       91.74%               38.05%
#                                            2. HMM VE:       90.03%               32.45%
#                                                             3. HMM MAP:       89.67%               39.20%
####

from collections import defaultdict
import math
import sys

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

# Set of all the tags
tag_set = {'adv', 'noun', 'adp', 'prt', 'det', 'num', '.', 'pron', 'verb', 'x', 'conj', 'adj'}

# Maximum value
max_val = - math.log(1.0 / 50000)


class Solver:
    """Class that contains all the logic to perform parts of speech tagging (POS tagging) for sentence
    """
    def __init__(self):
        """Constructor
        """
        # Dictionary that represents the initial probability table
        # Stores the negative logs of probabilities of starting with a particular part of speech
        # For example: {'noun': 1.234, 'verb': 2.234 ...}
        self.init_prob = dict()

        # Dictionary that stores the overall probabilities of POS tags
        # Stores the negative logs of probabilities of overall POS tags
        # For example: {'noun': 1.234, 'verb': 2.234 ...}
        self.tag_prob = dict()
        for tag in tag_set:
            self.tag_prob[tag] = 0

        # Dictionary of dictionaries that represent the transition probability table
        # Stores the negative logs of probabilities of transitioning from one part of speech to another
        # For example: {'noun': {'noun': 1.234, 'verb': 2.234 ..}, 'verb': {'noun': 1.234, 'verb': 2.234} ...}
        self.trans_prob = defaultdict(dict)
        for row_tag in tag_set:
            for col_tag in tag_set:
                self.trans_prob[row_tag][col_tag] = 0

        # Dictionary of dictionaries that represent the emission probability table
        # Stores the negative logs of probabilities of part of speech for a given word
        # For example: {'hanging': {'noun': 1.234, 'verb': 2.234 ..}, 'spider': {'noun': 1.234} ...}
        self.emit_prob = defaultdict(dict)

    @staticmethod
    def normalize_dict(dict_to_normalize):
        """Transforms count of a dictionaries to natural log of the probabilties
        :param dict_to_normalize: Dictionary that needs to be normalized
        :return:
        """
        total_log = math.log(sum(dict_to_normalize.values()))
        for key, val in dict_to_normalize.iteritems():
            dict_to_normalize[key] = sys.float_info.max if val == 0 else total_log - math.log(val)

    def posterior(self, sentence, label):
        """Calculate the log of the posterior probability of a given sentence with a given part-of-speech labeling
        :param sentence: List of words (string) for which posterior needs to be computed
        :param label:    List of labels (string) corresponding to the words in the sentence
        :return: Posterior probability (double)
        """
        post = 0
        for idx, word in enumerate(sentence):
            tag = label[idx]
            if tag in self.emit_prob[word]:
                if idx == 0:
                    post += self.init_prob[tag] + self.emit_prob[word][tag]
                else:
                    post += self.emit_prob[word][tag] + self.trans_prob[label[idx - 1]][tag]
        return -post

    def print_inputs(self):
        """Prints the dictionaries
        """
        def print_dict(dict_to_print, items_to_print):
            for key, val in dict_to_print.iteritems():
                items_to_print -= 1
                if items_to_print == 0:
                    break
                print 'Key ->', key, '   Val ->', val

        # print_inputs() starts from here
        print 'Printing initial probabilities', len(self.init_prob)
        print 'Size of initial probabilities', sys.getsizeof(self.init_prob)
        print_dict(self.init_prob, sys.maxint)

        print 'Printing tag probabilities', len(self.tag_prob)
        print 'Size of tag probabilities', sys.getsizeof(self.init_prob)
        print_dict(self.tag_prob, sys.maxint)

        print 'Printing transition probabilities', len(self.trans_prob)
        print 'Size of transition probabilities', sys.getsizeof(self.trans_prob)
        print_dict(self.trans_prob, sys.maxint)

        print 'Printing emission probabilities', len(self.emit_prob)
        print 'Size of emission probabilities', sys.getsizeof(self.emit_prob)
        print_dict(self.emit_prob, 50)

    def train(self, data):
        """Calculates the three probability tables for the given data
        :param data: List of 2 tuples [((sentence)(tags)), ((sentence)(tags))]
        """
        for sentence, tags in data:
            # Updating initial counts of tags in the initial dict
            if tags[0] not in self.init_prob:
                self.init_prob[tags[0]] = 0
            self.init_prob[tags[0]] += 1

            # Updating transition counts of tags in the transition dictionary
            for index in range(0, len(tags) - 1):
                curr_tag = tags[index]
                next_tag = tags[index + 1]
                self.trans_prob[curr_tag][next_tag] += 1

            # Updating tag and emission counts
            for word, tag in zip(sentence, tags):
                self.tag_prob[tag] += 1
                if tag not in self.emit_prob[word]:
                    self.emit_prob[word][tag] = 0
                self.emit_prob[word][tag] += 1

        # Normalizing initial probabilities table
        self.normalize_dict(self.init_prob)

        # Normalizing tag probabilities table
        self.normalize_dict(self.tag_prob)

        # Normalizing transition probabilities table
        for tag_dict in self.trans_prob.values():
            self.normalize_dict(tag_dict)

        # Normalizing emission probabilities table
        for tag_dict in self.emit_prob.values():
            for tag, count in tag_dict.iteritems():
                tag_dict[tag] = (self.tag_prob[tag] - math.log(count)) if count >= 1 else sys.float_info.max
            # self.normalize_dict(tag_dict)

    def simplified(self, sentence):
        """Returns most likely POS tags of words in a sentence
           Greedy approach
        :param sentence: List of words (string)
        :return: List of tags
        """
        # Initialize the list to store the tags
        predict_tag = [0] * len(sentence)
        for idx, word in enumerate(sentence):
            # Initialize minimum log probability as max float and tag as none
            min_log_prob = [sys.float_info.max, "noun"]
            # Find the minimum sum of emission and log probability of tag and pass the associated tag as our prediction
            for tag in self.emit_prob[word]:
                if idx == 0 and self.emit_prob[word][tag] + self.tag_prob[tag] < min_log_prob[0]:
                    min_log_prob = [self.emit_prob[word][tag] + self.tag_prob[tag], tag]
                elif self.emit_prob[word][tag] + self.tag_prob[tag] < min_log_prob[0]:
                    min_log_prob = [self.emit_prob[word][tag] + self.tag_prob[tag], tag]
            predict_tag[idx] = min_log_prob[1]
        return predict_tag

    def hmm_ve(self, sentence):
        """Returns most likely POS tags of words in a sentence
           by performing Variable Elimination algorithm
        :param sentence: List of words (string)
        :return: List of tags
        """
        tags = dict()  # To store all the tag probability values for all words
        # Calculating the probabilities for the first word
        tag_dict = dict()
        for tag in tag_set:
            if sentence[0] not in self.emit_prob.keys() or tag not in self.emit_prob[sentence[0]].keys():
                tag_dict[tag] = sys.float_info.max
            else:
                tag_dict[tag] = self.init_prob[tag] + self.emit_prob[sentence[0]][tag]
        tags[sentence[0]] = tag_dict

        # Calculating for the remaining words
        for w in range(1, len(sentence)):
            tag_dict = dict()
            prev_tags_dict = tags[sentence[w - 1]]

            for tag in tag_set:
                prob = sys.float_info.epsilon
                for prev_tag in prev_tags_dict:  # Scanning probabilities of the previous character
                    current_prob = prev_tags_dict[prev_tag] + self.trans_prob[prev_tag][tag]
                    prob += math.exp(-current_prob)  # For adding all the probabilities
                if sentence[w] not in self.emit_prob.keys() or tag not in self.emit_prob[sentence[w]].keys():
                    tag_dict[tag] = sys.float_info.max
                else:
                    tag_dict[tag] = self.emit_prob[sentence[w]][tag] - math.log(prob)
            tags[sentence[w]] = tag_dict
        return [tags[w].keys()[tags[w].values().index(min(tags[w].values()))] for w in sentence]

    def get_emission_probs(self, word):
        """Computes emission probabilities for a word
        :param word: string word
        :return: A dictionary mapping tag to the emission probability {'noun': 1.234, 'verb': 2.345) ...)
        """
        emission_prob_dict = dict()
        for tag in tag_set:
            if word in self.emit_prob and tag in self.emit_prob[word]:
                emission_prob_dict[tag] = self.emit_prob[word][tag]
            else:
                emission_prob_dict[tag] = max_val
        return emission_prob_dict

    def hmm_viterbi(self, sentence):
        """Returns most likely POS tags of words in a sentence
           by performing Viterbi algorithm
        :param sentence: List of words (string)
        :return: List of tags
        """
        tag_list = list(tag_set)  # Converting tag_set to a list to have indexes to refer
        rows = len(tag_list)
        cols = len(sentence)
        compatibility_matrix = [[None] * cols for i in range(rows)]

        # Storing a tuple in each cell (index of the previous cell, probability of the current cell)
        for col_index, curr_word in enumerate(sentence):
            curr_emission_probs = self.get_emission_probs(curr_word)
            for row_index, curr_tag in enumerate(tag_list):
                # Computing the probabilities for the first column
                if col_index == 0:
                    init_prob = self.init_prob[curr_tag] if curr_word in self.init_prob else max_val
                    compatibility_matrix[row_index][col_index] = (-1, curr_emission_probs[curr_tag] + init_prob)
                # Computing the probabilities of the other columns
                else:
                    best_prob_tuple = (-1, max_val)
                    for prev_row_index, prev_tag in enumerate(tag_list):
                        prev_prob = compatibility_matrix[prev_row_index][col_index - 1][1]
                        curr_prob = prev_prob + curr_emission_probs[curr_tag] + self.trans_prob[prev_tag][curr_tag]
                        if curr_prob < best_prob_tuple[1]:
                            best_prob_tuple = (prev_row_index, curr_prob)
                    compatibility_matrix[row_index][col_index] = best_prob_tuple

        # Backtracking to fetch the best path
        # Finding the cell with the max probability from the last column
        (max_index, max_prob) = (-1, max_val)
        for row in range(rows):
            curr_prob = compatibility_matrix[row][cols - 1][1]
            if curr_prob < max_prob:
                (max_index, max_prob) = (row, curr_prob)

        output_tag_list = list()  # List to store the output tags
        # Adding the best path to output list
        for col in range(cols - 1, 0, -1):
            output_tag_list.insert(0, tag_list[max_index])
            max_index = compatibility_matrix[max_index][col][0]
        output_tag_list.insert(0, tag_list[max_index])
        return output_tag_list

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        """This method is called by label.py, so you should keep the interface the
           It should return a list of part-of-speech labelings of the sentence
           one part of speech per word
        :param algo:     Algo that needs to be run to determine POS tags
        :param sentence: List of words (string)
        :return: List of tags
        """
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"
