import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

N = 5
potential_matrix = np.array([[np.e, 1], [1, np.e]])  # original potential matrix
# position_list
'''
[[ 0 10 20 30 40 50 60 70 80 90]
 [ 1 11 21 31 41 51 61 71 81 91]
 [ 2 12 22 32 42 52 62 72 82 92]
 [ 3 13 23 33 43 53 63 73 83 93]
 [ 4 14 24 34 44 54 64 74 84 94]
 [ 5 15 25 35 45 55 65 75 85 95]
 [ 6 16 26 36 46 56 66 76 86 96]
 [ 7 17 27 37 47 57 67 77 87 97]
 [ 8 18 28 38 48 58 68 78 88 98]
 [ 9 19 29 39 49 59 69 79 89 99]]
'''


class A_node(object):
    def __init__(self, id, row, col):
        self.id = id
        self.row = row
        self.col = col
        self.possible = [0, 1]


class Pair_potential(object):
    def __init__(self, idlist, potential_matrix):
        # idlist = [1,2]
        self.idlist = idlist
        self.potential_matrix = potential_matrix

    def query_id(self):
        return self.idlist

    def query(self, idlist_num):
        potential_matrix = self.potential_matrix
        for num in idlist_num:
            potential_matrix = potential_matrix[idlist_num[num]]
        return potential_matrix

    def combine_another_potential(self, Another_potential):
        Self_id = self.idlist
        Another_id = Another_potential.query_id()
        # assert Self_id[0] == Another_id[0] # eliminate one node each time

    def debug(self):
        print(self.idlist)


class the_bigX(object):
    def __init__(self, row, potential_list):
        start_id = row * N

        self.eli_id_list = list(range(start_id, start_id + N))
        self.qur_id_list = list(range(start_id + N, start_id + 2 * N))
        self.links = potential_list
        self.results = {}

        # print("bigX thing")

        eli_possibility_list = []
        qur_possibility_list = []
        for i in range(2 ** (N)):  # qur_possibility_list
            i_bin = binary_transfer(i, mode="X")
            qur_possibility_list.append(i_bin)
            eli_possibility_list.append(i_bin)

        for qur_list in tqdm(qur_possibility_list):
            count = 0
            for eli_list in eli_possibility_list:
                all_list = eli_list + qur_list  # len of 2N

                for link in self.links:  # all links query id only contain the 20 numbers
                    idlist = link.query_id()  # [0,1] [4,8]
                    # print("idlist",idlist)
                    log_potential_link = np.log(link.query(
                        idlist_num=[int(all_list[idlist[0] - start_id]), int(all_list[idlist[1] - start_id])]))
                    count = count + log_potential_link

            # print(qur_list)
            self.results.update({str(qur_list): count})

    def query_id(self, qur_list):
        return self.results[qur_list]

    def query_all(self):
        return list(self.results.values())


def binary_transfer(number, mode="S"):
    # a fast version
    # return a nice list
    if mode == "X":
        number_binary = (bin(number)[2:]).zfill(N)
    else:
        number_binary = (bin(number)[2:]).zfill(N * N)
    number_binary = list(number_binary)
    number_binary.reverse()
    return number_binary


class A_net(object):
    def __init__(self):
        self.position_list = np.resize(np.array(list(range(N ** 2))), new_shape=(N, N)).transpose()
        self.nodes = []
        self.links = []
        self.bigXs = []
        self.init_nodes()
        self.init_connections()
        self.init_bigX()
        # print(self.links[23].debug())
        # print(self.links[23].query([1,1]))

    def init_nodes(self):
        for node in range(N ** 2):
            row, col = np.asarray(self.position_list == node).nonzero()  # i1,j1 = row, col
            self.nodes.append(A_node(node, row, col))

    def init_connections(self):
        for s1 in range(N ** 2):
            i1, j1 = np.asarray(self.position_list == s1).nonzero()
            for s2 in range(s1 + 1, N ** 2):
                i2, j2 = np.asarray(self.position_list == s2).nonzero()
                if ((j1 == j2) and (abs(i1 - i2) == 1)) or ((i1 == i2) and (abs(j1 - j2) == 1)):
                    self.links.append(Pair_potential(idlist=[s1, s2], potential_matrix=potential_matrix))

    def init_bigX(self):
        for i in range(N):
            potential_list = []
            point_list = list(range(i * N, (i + 2) * (N)))
            # print(point_list)
            for link in self.links:
                idlist = link.query_id()
                if set(idlist).issubset(point_list):
                    potential_list.append(link)

            # print("potential_list", len(potential_list))
            big_X_i = the_bigX(i, potential_list)
            self.bigXs.append(big_X_i)

    def brute_force(self):

        possibility_list = []
        for i in range(2 ** (N * N)):
            i_bin = binary_transfer(i)
            possibility_list.append(i_bin)  # each is a (N*N) list
            count = 0

            for link in self.links:
                idlist = link.query_id()  # [1,2]
                # print(idlist)
                # print(i_bin)
                log_potential_link = np.log(link.query(idlist_num=[int(i_bin[idlist[0]]), int(i_bin[idlist[1]])]))
                count = count + log_potential_link

        return count

    def clever_algrithom_eachX(self):

        # print(len(self.bigXs))
        # bigX_last = self.bigXs[-2]
        # print(bigX_last.results)

        for big_X in self.bigXs:
            a = big_X.query_all()
            print(a)
            print(logsumexp(a))


Mynet = A_net()
print(Mynet.brute_force())
Mynet.clever_algrithom_eachX()

'''
for col_i in range(9): # col_i to eliminate
    col_j = col_i+1
    num_i = list(position_list[:,col_i])
    num_j = list(position_list[:,col_j])
    phi = phi_ori.copy()
    for eli_i in num_i:
        phi_keys = tuple([functools.reduce(lambda sub,ele: "{},{}".format(sub, ele), i[1:]) for i in phi.keys() if i[0] == eli_i])
        tem_list = []
        for phi_keys_i in phi_keys:
            if type(phi_keys_i) == int:
                tem_list.append(phi_keys_i)
            elif type(phi_keys_i) == str:
                phi_keys_i_part = [int(i) for i in phi_keys_i.split(",")]
                tem_list.extend(phi_keys_i_part)
        phi_keys = tuple(sorted(tem_list))

        print("phi_keys",phi_keys)
        new_potential_matrix = potential_matrix
        phi.update({phi_keys: new_potential_matrix})
'''
