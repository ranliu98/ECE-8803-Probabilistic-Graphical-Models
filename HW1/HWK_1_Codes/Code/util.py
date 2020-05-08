# -*- coding: utf-8 -*-
import sys
import re

RARE_TAG = '_RARE_'
RARE_WORD_THRESHOLD = 5


def is_numeric(word):
    find_numeric = False
    if re.match('[0-9]+', word):
        find_numeric = True

    return find_numeric


def is_all_uppercase(word):
    '''
    The word consists entirely of capitalized letters.
    '''
    all_uppercase = False
    if re.match('^[A-Z]+$', word):
        all_uppercase = True
    # for c in word:
    #     if not c.isupper():
    #         all_uppercase = False
    #         break
    return all_uppercase


def is_last_uppercase(word):
    last_uppercase = False
    if re.match('.*[A-Z]$', word):
        last_uppercase = True
    return last_uppercase


def rare_words_rule_p1(word):
    return RARE_TAG

def process_rare_words(input_file, output_file, rare_words, processer):
    """
    applying the rare word rule to process the training data
    """
    l = input_file.readline()
    while l:
        line = l.strip()
        if line:
            fields = line.split(' ')
            word = fields[0]
            tag = fields[-1]
            if word in rare_words:
                word = processer(word)  # applying rare word rule(s)
            output_file.write('{} {}\n'.format(word, tag))
        else:
            output_file.write('\n')
        l = input_file.readline()


def test_data_iterator(test_file):
    l = test_file.readline()
    while l:
        line = l.strip()
        if line:
            yield line  # a word
        else:
            yield None  # end of a line
        l = test_file.readline()


def test_sent_iterator(testdata_iterator):
    """
    return an iterator object that yields one sent at a time
    """
    current_sentence = []  # Buffer for the current sentence
    for word in testdata_iterator:
            if word is None:
                if current_sentence:  # Reached the end of a sentence
                    yield current_sentence
                    current_sentence = []  # Reset buffer
                else:  # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(word)  # Add token to the buffer

    if current_sentence:
        yield current_sentence



if __name__ == '__main__':
    test()
