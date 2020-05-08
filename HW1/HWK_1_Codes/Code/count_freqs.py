#! /usr/bin/python

__author__="Daniel Bauer <bauer@cs.columbia.edu>"
__date__ ="$Sep 12, 2011"

import sys
from collections import defaultdict
import math
import util
from decimal import Decimal

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    with open(corpus_file,"r") as a_file:
        lines = a_file.readlines()
    for l in lines:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)
        #l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in range(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        

def trigram_feature_reader(count_file_path = "rare_gene.counts"):
    with open(count_file_path,"r") as count_file:
        lines = count_file.readlines()
    # Two_GRAM_list = ["*_O","*_*","O_O","O_I-GENE","I-GENE_I-GENE","I-GENE_O","I-GENE_STOP","O_STOP","*_I-GENE"]
    Two_GRAM_dict = {}
    Three_GRAM_dict = {}
    for l in lines:
        line = l.strip()
        if line:
            fields = line.split(" ")
            if fields[1] == "2-GRAM":
                Two_GRAM_dict.update({"{}_{}".format(fields[2],fields[3]): int(fields[0])})
            elif fields[1] == "3-GRAM":
                Three_GRAM_dict.update({"{}_{}_{}".format(fields[2],fields[3],fields[4]): int(fields[0])})

    return Two_GRAM_dict, Three_GRAM_dict

def trigram_feature_counter(Two_GRAM_dict, Three_GRAM_dict):
    Two_GRAM_list = ["*_O", "*_*", "O_O", "O_I-GENE", "I-GENE_I-GENE", "I-GENE_O", "I-GENE_STOP", "O_STOP", "*_I-GENE"]
    Possible_list = ["_*","_O","_I-GENE","_STOP"]
    q_dict = {}
    for y_i in Possible_list:
        for y_i2_y_i1_ in Two_GRAM_list:
            Three_Gram_here = y_i2_y_i1_+y_i
            if Three_Gram_here in Three_GRAM_dict:
                q_dict.update({Three_Gram_here:Three_GRAM_dict[Three_Gram_here]/Two_GRAM_dict[y_i2_y_i1_]})
            else:
                q_dict.update({Three_Gram_here:0})

    print(q_dict)
    return q_dict


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in range(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count

class Hmm_ex(Hmm):
    """
    Store counts, and model params
    """
    def __init__(self, n=3):
        super(Hmm_ex, self).__init__(n)

        self.emission_params = defaultdict(float)  # e(x|y)
        self.word_counts = defaultdict(int)  # count(x)
        self.rare_words = []
        self.tags = set()
        self.words = set()
        self.q_3_gram = defaultdict(float)
        self.rare_word_threshold = util.RARE_WORD_THRESHOLD
        self.rare_words_rule = util.rare_words_rule_p1

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        super(Hmm_ex, self).train(corpus_file)
        self.count_words()  # count(x)
        self.get_rare_words(self.rare_word_threshold)  # rare word
        self.cal_emission_param()  # e(x|y)
        self.get_all_tags_and_words()  # get all tags and words
        self.cal_q_3_gram()  # q(y_i|y_i-2, y_i_1)

    def get_all_tags_and_words(self):
        for word, tag in self.emission_counts:
            self.tags.add(tag)
            self.words.add(word)

    def count_words(self):
        '''
        count(x)
        '''
        for word, tag in self.emission_counts:
            self.word_counts[word] += self.emission_counts[(word, tag)]

    def get_rare_words(self, threshold):
        self.rare_words = []
        for word in self.word_counts:
            if self.word_counts[word] < threshold:
                self.rare_words.append(word)

    def cal_emission_param(self):
        """
        e(x|y) = count(y->x) / count(y)
               = emission_counts[(y, x)] / ngram_counts[0][y]
        """
        for k in self.emission_counts.keys():
            # print k, self.emission_counts[k], tuple(k[-1:]), self.ngram_counts[0][tuple(k[-1:])]
            self.emission_params[k] = float(self.emission_counts[k]) / float(self.ngram_counts[0][tuple(k[-1:])])
            # print self.emission_params[k]

    def cal_q_3_gram(self):
        """
        q(y_i|y_i-2, y_i_1) = count(y_i-2, y_i-1, y_i) / count(y_i-2, y_i-1)
        """
        for c in self.ngram_counts[2]:
            self.q_3_gram[c] = float(self.ngram_counts[2][c]) / float(self.ngram_counts[1][tuple(c[0:2])])

    def read_counts(self, corpusfile):
        super(Hmm_ex, self).read_counts(corpusfile)

        self.word_counts = defaultdict(int)
        self.emission_params = defaultdict(float)
        self.rare_words = []
        self.q_3_gram = defaultdict(float)

        self.cal_emission_param()
        self.count_words()
        self.get_rare_words(self.rare_word_threshold)
        self.get_all_tags_and_words()
        self.cal_q_3_gram()

    def print_emission_counts(self, output):
        for word, ne_tag in self.emission_counts:
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))

    def print_ngram_counts(self, output, printngrams=[1, 2, 3]):
        for n in printngrams:
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" % (self.ngram_counts[n-1][ngram], n, ngramstr))

    def print_emission_params(self, output):
        for word, ne_tag in self.emission_params:
            output.write('{0} EMISSION_PARAM {1} {2}\n'.format(self.emission_params[(word, ne_tag)], ne_tag, word))

    def print_word_counts(self, output):
        for word in self.word_counts:
            output.write('{0} WORDCOUNT {1}\n'.format(self.word_counts[word], word))

    def print_rare_words(self, output):
        for word in self.rare_words:
            output.write('{0} RAREWORD {1}\n'.format(self.word_counts[word], word))

    def print_q_3_gram(self, output):
        for ngram in self.q_3_gram:
            output.write('{0} Q_3_GRAM {1} {2} {3}\n'.format(
                self.q_3_gram[ngram],
                ngram[0],
                ngram[1],
                ngram[2]))

class ViterbiTagger(Hmm_ex):
    """docstring for ViterbiTagger"""
    def __init__(self, n=3):
        super(ViterbiTagger, self).__init__(n)

    def tag(self, test_data_file, result_file):
        sent_iterator = util.test_sent_iterator(
            util.test_data_iterator(test_data_file))
        for sent in sent_iterator:
            tags = self.viterbi(sent)
            for s, t in zip(sent, tags):
                result_file.write('{0} {1}\n'.format(s, t))
            result_file.write('\n')

    def viterbi(self, sent):

        n = len(sent)
        pi = []
        bp = []
        for k in range(0, n + 1):
            pi.append(defaultdict(float))
            bp.append(defaultdict(str))
        pi[0][('*', '*')] = 1.0
        # decode
        for k in range(1, n + 1):
            W = self.tags
            U = self.tags
            V = self.tags
            x = sent[k - 1]
            if x in self.rare_words or x not in self.words:
                x = self.rare_words_rule(x)
            if k == 1:
                W = U = ('*',)
            if k == 2:
                W = ('*',)
            for u in U:
                for v in V:
                    max_pi, max_w = -1.0, ''
                    for w in W:
                        if self.emission_params[(x, v)] == 0.0:
                            continue
                        tmp = Decimal(pi[k - 1][(w, u)])  \
                            * Decimal(self.q_3_gram[(w, u, v)]) \
                            * Decimal(self.emission_params[(x, v)])
                        if tmp > max_pi:
                            max_pi, max_w = tmp, w
                    pi[k][(u, v)] = max_pi
                    bp[k][(u, v)] = max_w
        # trace back
        U = self.tags
        V = self.tags
        max_pi, max_u, max_v = -1.0, '', ''
        for u in U:
            for v in V:
                tmp = Decimal(pi[n][(u, v)]) \
                    * Decimal(self.q_3_gram[(u, v, 'STOP')])
                if tmp > max_pi:
                    max_pi, max_u, max_v = tmp, u, v
        result = list(range(0, n+1))
        result[n-1], result[n] = max_u, max_v
        for k in range(n-2, 0, -1):
            result[k] = bp[k+2][(result[k+1], result[k+2])]
        return result[1:]


def usage():
    print("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """)

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        with open(sys.argv[1],"r") as file:
            input = file
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n")
        sys.exit(1)

    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.train(sys.argv[1])
    # Write the counts
    counter.write_counts(sys.stdout)
