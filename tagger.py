import os
import sys
import argparse
import collections
import numpy as np

"""
1. Read the input files: Read the training and test files using Python.
 Use the provided starter code to help you parse the names of multiple
 training files, one test file, and one output file.

2. Preprocess the data: Split the text into sentences, tokenize the words
and tags, and handle ambiguity tags as mentioned in the assignment.

3. Calculate probabilities: Calculate the initial, transition, and observation
probabilities using the text-tag pairs in the training files.
a. Initial probabilities: For each POS tag, count the number of times it
appears at the beginning of a sentence and divide by the total
number of sentences.

b. Transition probabilities: For each POS tag, count the number of times
 it appears after another POS tag and divide by the total number of
 occurrences of the previous tag.

c. Observation probabilities: For each word-POS tag pair, count
the number of occurrences and divide by the total number of occurrences
of that POS tag.

4. Implement the Viterbi algorithm: Use the Viterbi
algorithm (or another suitable algorithm) to predict the
 POS tags for untagged text using the probabilities calculated in step 3.

5. Predict the POS tags for the test file: Run the Viterbi algorithm on
the test file to predict the POS tags for each word.

6. Handle unseen words: For words not present in the training files, you can
 consider strategies mentioned in the assignment, such as using the most
 likely POS tag given the word's position in the sentence or the previous word
 in the sentence.

7. Generate the output file: Create an output file with the predicted POS
tags in the same format as the training files.

8. Test your implementation: Test your program on the provided test cases
and any additional test cases you create. Ensure that it meets the
time constraint of 5 minutes for each test case.
"""

"""
# go to the training file, split into sentences, look at
# the first word in the sentence, look at the tag of the
# first word. How often do I see any(like verb) tag for
# the first word in a sentence.
I: initial probabilities    P(S_0)

# ex: If anywhere in a sentence I see a tag verb, then how
# likely I do see noun as the tag for the next word. (What is
# the likelyhood from verb to noun/noun to adjective)
T: transition matrix    P(S_k|S_k-1)

# suppose for a sentence, somewhere in this sentence is a tag
# verb, given this, how likely will be the result word in the
# training file
M: observation matrix  P(E_k|S_k)


Viterbi(E, S, I, T, M):
    # E = {E_0, E_1, E2, ..., E_t}, while E_0 will
    # be the first word in the sentence, E_1 is the
    # second word, ..., E_t will be the last word
    # in the sentence; Set of words

    # S = {S_0, S_1, S_2, ..., S_t}, while S is the
    set of tags in the sentence
    prob = matrix(length(E), length(S))
    prev = matrix(length(E), length(S))


# Determine values for time step 0
# Base case
for i in [0, ..., length(S) - 1]:
    prob[0, i] = I[i] * M[i, E[0]]      P(S_0) * P(E_0|S_0)
    prev[0, i] = None       it is as an flag. At the end, once we finish
                            all the calculations, and tracing back to find
                            the most likely sequnce of the tags, None is
                            for us to stop


# For time steps 1 to length(E) - 1
# find each current state's most likely prior
# state x.
# Recursive case
for t in [1, ..., length(E) - 1]:
    for i in [0, ..., length(S) - 1]:
        x = argmax_j in (prob[t - 1, j] * T[j, i] * M[i, E[t]])     argmax means the
                                                                    higher prob from previous
                                                                    state, and remember. j
                                                                    in this example
        prob[t, i] = prob[t - 1, x] * T[x, i] * M[i, E[t]]      remember the probability of most likely sequence
        prev[t, i] = x

return prob, prev
"""

All_POS_Tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS",
                "CJT", "CRD", "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1",
                "NN2", "NP0", "ORD", "PNI", "PNP", "PNQ", "PNX", "POS", "PRF",
                "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0", "UNC", 'VBB', 'VBD',
                'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI', 'VDN',
                'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB',
                'VVD', 'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0',
                'AJ0-VVN', 'AJ0-VVD', 'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP',
                'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0',
                'NN1-VVB', 'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0',
                'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
                'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1',
                'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

TAG_SIZE = len(All_POS_Tags)

# creating a dict to store the tag as the key with its position as value
tag_dict = {}
for i, tag in enumerate(All_POS_Tags):
    tag_dict[tag] = i

# the ending of a sentence
ENDING_PUN = ['!', '?', '.']


class HmmPos:
    """
    The class is created for a hidden markov model(HMM) that will handle with
    the input training and testing files for POS tagging problem.
    """

    def __init__(self, training_files, test_data):
        self.training_files = training_files
        self.test_data = test_data
        self.states = []  # A list ofPOS tags in the training file
        self.observations = []  # A list of words in the training file
        self.all_sentences = []  # all the sentences across the training file
        self.all_tag_lists = []  # all the tags at its correspond word pos
        self.initial_probs = np.zeros(TAG_SIZE)
        self.transition_probs = np.ones((TAG_SIZE, TAG_SIZE))
        self.observation_probs = np.zeros((TAG_SIZE, 0))
        self.unique_word_ref = {}

    def parsing_training_files(self):
        for file_name in self.training_files:
            with open(file_name, 'r', encoding='utf-8') as file:
                for line in file:
                    # Process the line to extract words and POS tags
                    line = line.split(':')
                    word = line[0].strip()
                    corr_tag = line[1].strip()
                    # Update the necessary data structure
                    self.observations.append(word)
                    self.states.append(corr_tag)

        return self.observations, self.states

    def splitting_sentences(self):
        for file_name in self.training_files:
            with open(file_name, 'r', encoding='utf-8') as file:
                for line in file:
                    char = []
                    for i, words in enumerate(line):
                        words = words.rstrip()
                        words = words.split(':')
                        words = []

    # def splitting_sentences(self):
    #     """
    #     splitting the training files into multiple sentences and its
    #     corresponding tags
    #     """
    #     sentence_list = []
    #     tag_list = []
    #     curr_sentence = []
    #     curr_tag_list = []
    #     for index, word in enumerate(self.observations):
    #         if word in ENDING_PUN:
    #             sentence_list.append(curr_sentence)
    #             curr_sentence = []
    #             tag_list.append(curr_tag_list)
    #             curr_tag_list = []
    #         else:
    #             curr_sentence.append(word)
    #             curr_tag_list.append(self.states[index])
    #     self.all_sentences = sentence_list
    #     self.all_tag_lists = tag_list
    #
    #     return self.all_sentences, self.all_tag_lists

    def initial_prob_training(self):
        """
        how likely each POS tag appears at the beginning of a sentence
        """
        sentence_size = len(self.all_tag_lists)
        for index in range(sentence_size):
            first_tag = self.all_tag_lists[index][0]
            tag_index_in_dict = tag_dict[first_tag]
            self.initial_probs[tag_index_in_dict] += 1
        self.initial_probs /= sentence_size

        return self.initial_probs

    def transition_prob_training(self):
        """
        how likely each POS tag from one tag to another.
        """
        prev_count = np.zeros(TAG_SIZE)
        for sentence in self.all_sentences:
            for i in range(1, len(sentence)):
                current = sentence[i][1]
                prior = sentence[i - 1][1]
                row_idx = tag_dict[prior]
                col_idx = tag_dict[current]
                self.transition_probs[row_idx][col_idx] += 1
                prev_count[row_idx] += 1

        for r, count in enumerate(prev_count):
            if count != 0:
                self.transition_probs[r] /= (count + TAG_SIZE)

        return self.transition_probs

    def observation_prob_training(self):
        """
        how likely each POS tag to each observed word.
        """
        flag = np.zeros(TAG_SIZE)
        accumulator = 0

        word_len = len(self.observations)
        for index in range(word_len):
            curr_word = self.observations[index]
            curr_tag = self.states[index]
            flag[tag_dict[curr_tag]] += 1
            if curr_word not in self.unique_word_ref:
                self.unique_word_ref[curr_word] = accumulator
                accumulator += 1
                self.observation_probs = np.hstack(
                    (self.observation_probs, np.zeros((TAG_SIZE, 1))))
            r = tag_dict[curr_tag]
            c = self.unique_word_ref[curr_word]
            self.observation_probs[r][c] += 1

        for r, count in enumerate(flag):
            if count != 0:
                self.observation_probs[r] /= count

        return self.observation_probs, self.unique_word_ref




# wrong, needs to be updated
def viterbi_algo(E, S, I, T, M, unique_words):
    prob = np.zeros(len(E), len(S))
    prev = np.zeros(len(E), len(S))

    for index in range(len(S)):
        if E[0] not in unique_words:
            prob[0, index] = I[index]
            prev[0, index] = None
        else:
            prob[0, index] = I[index] * M[index, unique_words[E[0]]]
            prev[0, index] = None

    for t in range(1, len(E)):
        for index in range(len(S)):
            if E[t] not in unique_words:
                x = np.argmax(prob[t - 1, :] * T[:, index])
                prob[t, index] = prob[t - 1, x] * T[x, index]
                prev[t, index] = x
            else:
                x = np.argmax(prob[t - 1, :] * T[:, index] *
                              M[index, unique_words[E[t]]])
                prob[t, index] = prob[t - 1, x] * T[x, index] * \
                                 M[index, unique_words[E[t]]]
                prev[t, index] = x
    return prob, prev


def output_file_tags(prob, prev, words):
    result_tag = ""
    result_lst = []
    index = np.argmax[prob[-1]]
    flag = [index]
    for i in range(len(prob) - 1, 0, -1):
        curr_tag_index = flag[0]
        result_lst.insert(0, All_POS_Tags[curr_tag_index])
        prev_tag = prev[i][curr_tag_index]
        flag.insert(0, int(prev_tag))
        s1 = "{} : {}\n".format(words[i], All_POS_Tags[curr_tag_index])
        result_tag += s1

    curr_tag_index = flag[0]
    result_lst.insert(0, All_POS_Tags[curr_tag_index])
    s2 = "{} : {}\n".format(words[0], All_POS_Tags[curr_tag_index])
    result_tag += s2

    return result_tag


def write_solution(solString, outputFile):
    output = open(outputFile, 'w')
    output.write(solString)
    output.close()
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    print("Starting the tagging process.")

    # start processing the files
    for trainingFile in training_list:
        f = open(trainingFile)
        lines = f.readlines()

        s = []
        all_sentences = []
        for i, l in enumerate(lines):
            l = l.rstrip()
            # base case: the word is a colon
            if l[0] == ':':
                l = l.rstrip()
                word = l[0]  # the colon word
                l = l[1:]  # without the colon
                l = l.split(":")
                l[0] = word
                l = [x.strip() for x in l]
            else:
                l = l.split(':')
                l = [x.strip() for x in l]
            lines[i] = l
            s.append(l)
            if l[0] in ['.', '?', '!', '...']:
                all_sentences.append(s)
                s = []
        if s != []:
            all_sentences.append(s)

    hmm = HmmPos(training_list, args.testfile)
    t_observations, t_states = hmm.parsing_training_files()
    t_all_sentence, t_all_tags = hmm.splitting_sentences()
    initial_prob = hmm.initial_prob_training()
    transition_prob = hmm.transition_prob_training(all_sentences)
    observation_prob, unique_words_lst = hmm.observation_prob_training()

    result = ""
    for sentence in t_all_sentence:
        prob, prev = viterbi_algo(sentence, All_POS_Tags, initial_prob,
                                  transition_prob, observation_prob,
                                  unique_words_lst)
        test_line = output_file_tags(prob, prev, sentence)
        result += test_line

    write_solution(result, args.outputfile)


def fbi(n):
    if n == 1 or n == 2:
        result = 1
    else:
        ref = "11"
        index = 2
ç            num = int(ref[n-1]) + int(ref[n-1])
            ref += str(num)
            index += 1

        result = int(ref[n-1])

    return result



