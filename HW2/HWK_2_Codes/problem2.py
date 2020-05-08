import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy.io import savemat

np.set_printoptions(threshold=sys.maxsize)


def test_read_data():
    HW2_data_file = os.path.join("data-HWK2-2020", "Data-Problem-2", "noisystring.mat")
    data = sio.loadmat(HW2_data_file)
    data = data["noisystring"].tolist()[0]

    first_name = ["david", "anton", "fred", "jim", "barry"]
    sur_name = ["barber", "ilsung", "fox", "chain", "fitzwilliam", "quinceadams", "grafvonunterhosen"]

    pattern_list = [re.compile(i) for i in first_name + sur_name]

    data_test = "davidsadjhfkdasbarber"

    for i, pattern in enumerate(pattern_list):
        data = re.sub(pattern, str(i), data)

    print(data)


# test_read_data()

def alphabet_list():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    chara_list = []
    for chara_i in range(len(alphabet)):
        chara_list.append(alphabet[chara_i])
    return chara_list


def convert_name(some_name):
    chara_list = alphabet_list()
    number_list = []
    for i in range(len(some_name)):
        number_list.append(chara_list.index(some_name[i]))
    assert len(number_list) == len(some_name)
    return number_list


def load_data():
    HW2_data_file = os.path.join("data-HWK2-2020", "Data-Problem-2", "noisystring.mat")
    data = sio.loadmat(HW2_data_file)
    data = data["noisystring"].tolist()[0]

    return convert_name(data)


class MyHmm(object):  # base HMM
    def __init__(self, transition_probs, emission_probs, initial_distribution):
        # A = Transition probs, B = Emission Probs, pi = initial distribution
        #
        self.A = transition_probs
        self.states = list(range(26))  # get the list of states
        self.N = len(self.states)  # number of states of the model
        self.B = emission_probs
        self.pi = initial_distribution
        return

    def forward(self, obs):
        self.fwd = [{}]
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob

    def viterbi(self, obs):
        vit = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]

        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max((vit[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
                # Don't need to remember the old paths
            path = newpath
        n = 0  # if only one element is observed max is sought in the initialization values
        if len(obs) != 1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return prob, path[state]


def bp_search(observations, transition_probs, emission_probs, prior):
    HMM = MyHmm(transition_probs=transition_probs, emission_probs=emission_probs, initial_distribution=prior)
    loglik = HMM.forward(observations)
    print(loglik)
    print("oh HMM learns something here")
    sigma, logpvhstar = HMM.viterbi(observations)
    print("oh HMM learns something other here")
    return sigma, loglik, logpvhstar


def algorithm_sp(pattern, patternstate, generalstate, tran, corpt, other_general, prior):
    # return [spphghm, pvgh, ph1, startpattern, generalstates]
    pvgh_pattern = {}
    for pt in patternstate:
        for l in range(0, len(pattern[pt])):
            pvgh_pattern["{}_{}".format(pt, l)] = corpt[:, pattern[pt][l]]

    startpattern = np.zeros(shape=len(pattern))
    endpattern = np.zeros(shape=len(pattern))

    startpattern[0] = 1
    endpattern[0] = len(pattern[1])
    for p in range(1, len(pattern)):
        startpattern[p] = endpattern[p - 1] + 1
        endpattern[p] = startpattern[p] + len(pattern[p]) - 1
    H = endpattern[-1] + len(generalstate)
    H = int(round(H))

    phghm = np.zeros(shape=(H, H))
    Gstart = int(round(endpattern[-1] + 1))
    generalstates = list(range(Gstart, (Gstart + len(generalstate) - 1)))

    # setup the prior (distribution of initial time):
    ph1 = np.zeros(shape=(H, 1))
    if np.sum(prior) == 0:
        ph1 = np.ones(shape=(H, 1)) / H  # uniform if ph1==0
    if 'generalstate' not in prior and 'patternstate' not in prior:
        ph1 = prior
    if 'generalstate' in prior:
        for gt in range(len(generalstate)):
            ph1[Gstart + gt - 1, 0] = prior["generalstate"][gt]
    if 'patternstate' in prior:
        for pt in patternstate:
            if "startatbeginningpattern" in prior:
                ph1[startpattern[pt], 1] = prior["patternstate"][pt]  # % start at pattern
            else:
                ph1[startpattern[pt]:endpattern[pt], 1] = prior["patternstate"][pt] / len(
                    pattern[pt])  # % start at pattern

    for ptm in patternstate:
        for pt in patternstate:
            phghm[int(startpattern[pt]), int(endpattern[ptm])] = tran[int(pt), int(ptm)]
            # print(phghm[int(startpattern[pt]),int(endpattern[ptm])])
    for pt in range(len(patternstate)):
        s = int(startpattern[pt])
        for l in range(len(pattern[pt]) - 1):
            phghm[s + 1, s] = 1
            s = s + 1
    for pt in patternstate:
        for g in range(len(generalstate)):
            phghm[int(startpattern[pt]), Gstart + g - 1] = tran[int(pt), generalstate[g]]
    for pt in patternstate:
        for g in range(len(generalstate)):
            phghm[Gstart + g - 1, int(endpattern[pt])] = tran[generalstate[g], int(pt)]
    for gt in range(len(generalstate)):
        for gtm in range(len(generalstate)):
            phghm[Gstart + gt - 1, Gstart + gtm - 1] = tran[generalstate[gt], generalstate[gtm]]
    pvgh = np.zeros(shape=(26, H))
    for pt in patternstate:  # 11
        for l in range(len(pattern[pt])):
            pvgh[:, int(startpattern[pt]) + l - 1] = pvgh_pattern["{}_{}".format(pt, l)]
    for g in range(len(generalstate)):
        pvgh[:, Gstart + g - 1] = other_general[:, g]

    return phghm, pvgh, ph1, startpattern, generalstates


def regetp(p, startIDX, genIDX):
    # return [pnum gnum]
    pnum = []
    gnum = []
    for i in range(len(p)):
        try:
            k = np.asarray(p == startIDX).nonzero()
        except:
            k = False
        try:
            l = np.asarray(p == genIDX).nonzero()
        except:
            l = False

        if k:
            pnum.append(k)
        else:
            pnum.append("Nan")
        if l:
            gnum.append(l)
        else:
            gnum.append("Nan")
        if not k and not l:
            pnum[i] = pnum[i - 1]
            gnum[i] = gnum[i - 1]
    return pnum, gnum


first_name = ["david", "anton", "fred", "jim", "barry"]
sur_name = ["barber", "ilsung", "fox", "chain", "fitzwilliam", "quinceadams", "grafvonunterhosen"]

first_name_pattern = [convert_name(name) for name in first_name]
sur_name_pattern = [convert_name(name) for name in sur_name]
pattern = first_name_pattern + sur_name_pattern


def transistion_matrix():
    transistion_m = np.zeros(shape=(15, 15))  # 12 names plus 3 general states
    transistion_m[12, 12] = 0.8
    transistion_m[13, 13] = 0.8
    transistion_m[12, 14] = 1
    for p in range(0, 12):
        if p < 5:
            transistion_m[p, 12] = 0.2 / 5
            transistion_m[13, p] = 1
        elif p >= 5:
            transistion_m[p, 13] = 0.2 / 7
            transistion_m[14, p] = 1
    return transistion_m


# emission:
emission_p = 0.3
corptProb = np.ones(shape=(26, 26)) * (1 - emission_p) / 25
for i in range(26):
    corptProb[i, i] = emission_p

other_general = np.ones(shape=(26, 3)) / 26
prior = {}
prior["generalstate"] = np.array([1, 0, 0]).transpose()

transistion_m = transistion_matrix()

# print(transistion_m)

phghm, pvgh, ph1, st_pat_list, ge_pat_list = \
    algorithm_sp(pattern, list(range(12)), list(range(12, 14)), transistion_m, corptProb, other_general, prior)

# print(startpatternIDX)
# print(generalstateIDX)

savemat('prob2inter.mat', dict(trans=phghm, emiss=pvgh, prior=ph1))

# print(phghm.shape)
# print(pvgh)

# print(startpatternIDX)
# print(endpatternIDX)

print("finish pattern setup")

observed_vec = load_data()
sigma, loglik, loglikpvhstar = bp_search(observed_vec, phghm, pvgh, ph1)

print("finish pattern search")

print(sigma)
print(loglik)
print(loglikpvhstar)

pnum, gnum = regetp(sigma, st_pat_list, ge_pat_list)
pcount = np.zeros(shape=(12, 12))

# print(pnum)
# print(gnum)

fname = 0
sname = 0
fnamep = -1
snamep = -1

for i in range(len(observed_vec)):
    if pnum[i] > 0:
        if gnum[i - 1] == 1:
            fname = fname + 1
            fnamep = pnum[i]
        if gnum[i - 1] == 2:
            sname = sname + 1
            snamep = pnum[i]
        if fname + sname == 2:
            pcount[fnamep, snamep] = pcount[fnamep, snamep] + 1
            fname = 0
            sname = 0
