from collections import defaultdict
from tqdm import tqdm
from count_freqs import trigram_feature_counter, trigram_feature_reader, simple_conll_corpus_iterator, sentence_iterator
from count_freqs import ViterbiTagger
import util
"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

with open("/Users/ranliu/Desktop/Class-related/20Spring/ECE8803/HW/hmm-data/gene.counts", "r") as orgi_file:
    lines = orgi_file.readlines()


def get_rare_word_dict(lines):
    word_dict = {}
    for l in lines:
        line = l.strip()
        if line:
            fields = line.split(" ")
            if fields[1] == "WORDTAG":
                try:
                    if int(fields[0]) < 5:
                        word_dict.update({fields[3]: (fields[0], fields[2])})
                except:
                    print("Error in ", fields[0])
    return word_dict


def check_if_word_list_correct(word_dict, lines):
    wrong_word_dict = {}
    for l in lines:
        line = l.strip()
        if line:
            fields = line.split(" ")
            if fields[1] == "WORDTAG":
                if fields[3] in word_dict.keys():
                    temp = word_dict[fields[3]]
                    if temp != (fields[0], fields[2]):
                        # print("Error here")
                        # print("temp is {} but read {}".format(temp, (fields[0],fields[2])))
                        if int(fields[0]) + int(temp[0]) >= 5:
                            wrong_word_dict.update({fields[3]: int(fields[0]) + int(temp[0])})

    return wrong_word_dict


def get_final_rare_list(write_out=False):
    word_dict = get_rare_word_dict(lines)
    print(len(word_dict))
    wrong_word_dict = check_if_word_list_correct(word_dict, lines)
    print(len(wrong_word_dict))
    final_rare_word_list = [n for n in word_dict.keys() if n not in wrong_word_dict.keys()]
    print(len(final_rare_word_list))
    print(final_rare_word_list)

    if write_out:
        with open("rare_word_list.txt", "w") as write_file:
            for word in final_rare_word_list:
                write_file.write(str(word))
                write_file.write("###---###")

    return final_rare_word_list


def get_rare_gene_train():
    final_rare_word_list = get_final_rare_list()
    with open("rare_gene.train", "w") as train_file_with_rare:

        with open("gene.train", "r") as orginal_data_file:
            org_lines = orginal_data_file.readlines()
        for line in tqdm(org_lines):
            line = line.strip()
            if line:
                line = line.split(" ")
                if line[0] in final_rare_word_list:
                    line[0] = "_RARE_"
                train_file_with_rare.write("{} {}\n".format(line[0], line[1]))

            else:
                train_file_with_rare.write("\n")


def emission(I_GENE_dict, NO_GENE_dict):
    # the emission function
    # input: {"word": (times, tagger)}
    # output: {"word": (emission_I, emission_no)}

    I_GENE_list = I_GENE_dict.keys()
    NO_GENE_list = NO_GENE_dict.keys()

    I_only_list = [n for n in I_GENE_list if n not in NO_GENE_list]
    NO_only_list = [n for n in NO_GENE_list if n not in I_GENE_list]
    Both_list = [n for n in I_GENE_list if n in NO_GENE_list]
    assert len(Both_list) + len(I_only_list) == len(I_GENE_list)
    assert len(Both_list) + len(NO_only_list) == len(NO_GENE_list)

    I_GENE_total_number_list = [int(times) for times, tagger in I_GENE_dict.values()]
    I_GENE_total = sum(I_GENE_total_number_list)

    NO_GENE_total_number_list = [int(times) for times, tagger in NO_GENE_dict.values()]
    NO_GENE_total = sum(NO_GENE_total_number_list)

    # I divided words into three categories due to unknown reason
    # but luckily it does work
    # sorry if that is difficult to read >_<
    for words in I_only_list:
        yield {words: (int(I_GENE_dict[words][0]) / I_GENE_total, 0)}
    for words in NO_only_list:
        yield {words: (0, int(NO_GENE_dict[words][0]) / NO_GENE_total)}
    for words in Both_list:
        yield {words: (int(I_GENE_dict[words][0]) / I_GENE_total, int(NO_GENE_dict[words][0]) / NO_GENE_total)}


def get_emission_result_for_rare_count_f():
    # result sample: {'Ag': (0.1, 0.2)}
    with open("rare_gene.counts", "r") as rare_count_f:
        lines = rare_count_f.readlines()

    I_GENE_dict = {}
    NO_GENE_dict = {}

    for line in lines:
        l = line.strip()
        if l:
            parts = l.split(" ")
            if parts[1] == "WORDTAG":
                if parts[2] == "O":
                    NO_GENE_dict.update({parts[3]: (parts[0], parts[2])})
                if parts[2] == "I-GENE":
                    I_GENE_dict.update({parts[3]: (parts[0], parts[2])})

    result = emission(I_GENE_dict, NO_GENE_dict)

    return result


def compare_tuple(a_tuple):
    if a_tuple[0] > a_tuple[1]:
        temp = "I-GENE"
    else:
        temp = "O"
    return temp


def baseline_usage():
    result = get_emission_result_for_rare_count_f()
    result_dict = {}
    [result_dict.update(n) for n in result]

    print(result_dict)
    print(len(result_dict), type(result_dict))

    with open("gene_test.p1.out", "w") as gene_test_p1_out:

        with open("/Users/ranliu/Desktop/Class-related/20Spring/ECE8803/HW/hmm-data/gene.test", "r") as test_file:
            lines = test_file.readlines()

        for line in tqdm(lines):
            word = line.strip()
            if word:
                if word in result_dict.keys():
                    temp = compare_tuple(result_dict[word])
                    gene_test_p1_out.write("{} {}\n".format(word, temp))
                else:
                    temp = compare_tuple(result_dict["_RARE_"])
                    gene_test_p1_out.write("{} {}\n".format(word, temp))
            else:
                gene_test_p1_out.write("\n")


# Use the below command line to produce baseline model
# baseline_usage()
#
########################################
# Trigram model
########################################

'''
def add_boundary(sent_iterator):
    for sent in sent_iterator:
        # Add boundary symbols to the sentence
        w_boundary = 2 * [(None, "*")]
        w_boundary.extend(sent)
        w_boundary.append((None, "STOP"))
        yield w_boundary

def pi_viterbi_k1(k, u, v, q_funct, emission_result):
    # u: S(k-1) -- None, "*"
    # v: S(k)
    assert k == 1
    r_O = q_funct_with_artificial_trigram(q_funct, trigram_part0="*", trigram_part1="*", trigram_part2="O") * \
          emission_result[v][1]
    r_I = q_funct_with_artificial_trigram(q_funct, trigram_part0="*", trigram_part1="*", trigram_part2="I-GENE") * \
          emission_result[v][0]
    if r_O > r_I:
        result = ("O", r_O)
    else:
        result = ("I-GENE", r_I)
    return result

def pi_viterbi_rec(k, u, v, q_funct, emission_result):
    if k > 2:
        ...

def count_r_k(sent, k, n, emission_result, q_funct):
    # q_funct example: '*_*_O': 0.9457089011307626
    # emission_result example: {"word": (emission_I, emission_O)}
    assert k <= n and k >= 1
    sent = sent[:k + 2]

    tag_q_list = [tag for word, tag in sent]
    emission_list = [emission_result[word] if word in emission_result else emission_result["_RARE_"] for word, tag in
                     sent]
    print(emission_result["_RARE_"])

    # for number in range(k):
    #    r = q_funct["{}_{}_{}".format(tag_q_list[number],tag_q_list[number+1],tag_q_list[number+2])] \
    #        * emission_list[number+2]
    # pi_k_u_v(k, u, v) = max(r("*_*_O"),r("*_*_I-GENE"))
'''

###


def q_funct_with_artificial_trigram(q_funct, trigram_part0, trigram_part1, trigram_part2):
    # sentence trigram_part0 + trigram_part1 + trigram_part2
    return q_funct["{}_{}_{}".format(trigram_part0, trigram_part1, trigram_part2)]

def viterbi(sentence, e_result_dict, q_funct):
    def findSet(index):
        if index in range(1, len(sentence) + 1):
            return {"O", "I-GENE"}
        elif index == 0 or index == -1:
            return {"*"}
        elif index == len(sentence) + 1:
            return {"STOP"}

    def pi_viterbi(k, u, v, sentence, e_result_dict, q_funct):
        # return ((2, 'O', 'O'), 4.849205978634416e-10)

        prob = defaultdict(float)
        if k == 0 and u == '*' and v == '*':
            return (1., '*')
        else:
            for w in findSet(k - 2):
                prev = pi_viterbi(k - 1, w, u, sentence, e_result_dict, q_funct)[0]
                # tuple((w,u,v))
                q = q_funct_with_artificial_trigram(q_funct, w, u, v)  # [((w, u, v))]
                if v == "O":
                    if sentence[k - 1][1] in e_result_dict:
                        e = float(e_result_dict[sentence[k - 1][1]][1])  # if k==1, return sentence(0)
                    else:
                        e = float(e_result_dict["_RARE_"][1])
                elif v == "I-GENE":
                    if sentence[k - 1][1] in e_result_dict:
                        e = float(e_result_dict[sentence[k - 1][1]][0])
                    else:
                        e = float(e_result_dict["_RARE_"][0])
                probability = prev * q * e
                prob[tuple((w, u))] = probability
            max_tuple = max(prob.items(), key=lambda x: x[1])

            return max_tuple[1], max_tuple[0][0]

    backpointer = defaultdict(str)
    tags = defaultdict(str)
    k = len(sentence)


    for i in tqdm(range(1, k + 1)):
        prob = defaultdict(float)
        for u in findSet(i - 1):
            for v in findSet(i):
                value, w = pi_viterbi(i, u, v, sentence, e_result_dict, q_funct)
                prob[tuple((i, u, v))] = value
                #backpointer[tuple((i, u, v))] = w
                #print("i-{} ### u-{},v-{},value-{}".format(i,u,v,value))
                #print("bp in middle tuple((i, u, v))-{} w-{}".format(tuple((i, u, v)), w))
        max_tuple = max(prob.items(), key=lambda x: x[1])  # max P{ (u,v)set }
        backpointer[tuple((i, max_tuple[0][1], max_tuple[0][-1]))] = max_tuple[0][1]  # bp (k,u,v)= tag w
        # print("bp in final tuple((i, max_tuple[0][1], max_tuple[0][-1]))-{} value-{}".format(tuple((i, max_tuple[0][1], max_tuple[0][-1])), max_tuple[0][1]))
        # sentence_with_tag.append(max_tuple[0][-1])

    for i in range((k - 2), 0, -1):
        tag = backpointer[tuple(((i + 2), tags[i + 1], tags[i + 2]))]
        tags[i] = tag

    tag_list = list()
    for i in range(1, len(tags) + 1):
        tag_list.append(tags[i])

    return tag_list

def viterbi_more(corpus_file="gene.test"):
    wb_sent_iterator = (sentence_iterator(simple_conll_corpus_iterator(corpus_file)))

    Two_GRAM_dict, Three_GRAM_dict = trigram_feature_reader()
    q_funct = trigram_feature_counter(Two_GRAM_dict, Three_GRAM_dict)
    emission_result = get_emission_result_for_rare_count_f()
    e_result_dict = {}
    [e_result_dict.update(n) for n in emission_result]

    for sent in wb_sent_iterator:
        print(sent)
        tag_list = viterbi(sent, e_result_dict, q_funct)
        print(tag_list)

def vitergi_hmm(test_data_filename, result_filename, hmm_model_filename):

    tagger = ViterbiTagger(3)
    tagger.read_counts(open(hmm_model_filename,"r"))

    tagger.tag(open(test_data_filename,"r"), open(result_filename, 'w'))

def viterbi_usage():
    vitergi_hmm('gene.test', 'gene_test.p2.out', 'rare_gene.counts')

viterbi_usage()
