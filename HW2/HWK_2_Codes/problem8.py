import numpy as np
import os
import scipy.io as sio
import plotly.graph_objects as go
import time


HW8_data_file_dir = os.path.join("Data-Part-B","Data-Problem-8")
Dataset_file = os.path.join(HW8_data_file_dir,"dataset.dat")
joint_file = os.path.join(HW8_data_file_dir,"joint.dat")

# assign numbers sequ
name_of_nodes = ["IsSummer", "HasFlu", "HasFoodPoisoning", "HasHayFever", "HasPneumonia", "HasRespiratoryProblems",
                 "HasGastricProblems", "HasRash", "Coughs", "IsFatigued", "Vomits", "HasFever"]

def list1234_sp():
    feed_list = list(range(16))
    def list1234_sp_inside(number):
        result_list = []
        while number != 0:
            number_new = number//2
            if number_new*2 == number:
                result_list.append(0)
            else:
                result_list.append(1)
            number = number_new
        while len(result_list) != 4:
            result_list.append(0)
        return result_list
    return [list1234_sp_inside(i) for i in feed_list]

def number2list(number):
    result_list = []
    while number != 0:
        number_new = number//2
        if number_new*2 == number:
            result_list.append(0)
        else:
            result_list.append(1)
        number = number_new
    while len(result_list) != 12:
        result_list.append(0)
    return result_list

def prepare_dataset():
    with open(Dataset_file,"r") as dataset_file:
        lines = dataset_file.readlines()
    lines = [number2list(int(line.strip())) for line in lines]
    lines = np.array(lines)
    print("here you have the dataset, shape is {}".format(lines.shape))
    return lines

def prepare_joint():
    with open(joint_file, "r") as j_file:
        lines = j_file.readlines()
    joint_dis = {}
    for line in lines:
        line = line.strip().split("\t")
        joint_dis[(int(line[0]))] = float(line[1])

    return joint_dis

class node_distribution_table(object):
    def __init__(self, node_number_list, condition_list):
        # if node_number_list is a number, return just one table
        # if node_number_list is a list, return all
        # condition_list is np_array
        self.nodes = node_number_list
        self.condition_list = condition_list
        for node in node_number_list:
            self.assign_type(node)
            self.count_prob(node)

    def assign_type(self, node):
        if node == 0:
            self.node_type = 0
        elif node in list(range(1,5)):
            self.node_type = 1
        elif node in list(range(5,12)):
            self.node_type = 2

    def count_prob(self, node):
        if self.node_type == 0:
            # True, False table
            # num1, num2
            all_sum = np.sum(self.condition_list, axis=0)
            print(all_sum.shape)
            p_type0_T = all_sum[0]
        elif self.node_type == 1:
            # 1 \ 0 True False table
            # True  num1 num2
            # False num3 num4
            condition_list = self.condition_list
            num1_num3 = condition_list[condition_list[:,0]==1]
            num2_num4 = condition_list[condition_list[:,0]==0]
            num1 = num1_num3[num1_num3[:,node]==1]
            num3 = num1_num3[num1_num3[:,node]==0]
            num2 = num2_num4[num2_num4[:,node]==1]
            num4 = num2_num4[num2_num4[:,node]==0]

        elif self.node_type == 2:
            # 1,2,3,4 \ 5 True False
            # T T T T     num1 num2
            # F T T T     num3 num4
            # T F T T     num5 num6
            # T T F T     num7 num8
            # T T T F     num9 num10
            # F F T T     num11 num12
            # F T F T     num13 num14
            # F T T F     num15 num16
            # T F F T     num17 num18
            # T F T F     num19 num20
            # T T F F     num21 num22
            # F F F T     num23 num24
            # F F T F     num25 num26
            # F T F F     num27 num28
            # T F F F     num29 num30
            # F F F F     num31 num32
            cl = self.condition_list
            num_list = []
            list1234oflist = list1234_sp()
            for list1234 in list1234oflist:
                for tf5 in range(2):
                    num = cl[(cl[:,1]==list1234[0]) & (cl[:,2]==list1234[1]) & (cl[:,3]==list1234[2]) & (cl[:,4]==list1234[3])
                             & (cl[:,node]==tf5)]
                    num_list.append(num)

def draw_table():

    fig = go.Figure(data=[go.Table(
        header=dict(values=['A Scores', 'B Scores'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[[100, 90, 80, 90], # 1st column
                           [95, 85, 75, 95]], # 2nd column
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align='left'))
    ])

    fig.update_layout(width=500, height=300)
    fig.show()

class A_node(object):

    def __init__(self, Id, values, parents, children, prob, p_back):
        self.Id = Id
        self.values = values
        self.parents = parents
        self.children = children
        self.prob = prob
        self.prob_back = p_back

class Bayesian_Model(object):
    def func(self, adj, Cpt):
        adj_trans= adj.transpose()
        root = []
        for i_xx in range(0,adj.shape[0]):
            if sum(adj_trans[i_xx])==0:
                root.append(i_xx)
        group = []
        for i in range(0,adj.shape[0]):
            numid=i
            values= 0
            parents= np.nonzero(adj[:,i])
            parents= parents[0].tolist()
            children = np.nonzero(adj[i,:])
            children = children[0].tolist()
            cpt=Cpt[i]
            group.append(A_node(numid,values,parents,children,cpt,None))
        return group,root

    def accuracy_check(self, group, root):
        diff = []

        with open(os.path.join("Data-Part-B","Data-Problem-8","joint.dat"), 'r') as joint_f:
            d = joint_f.readlines()
        for line in d:
            k = line.strip().split('\t')
            i_bin = binary_transfer(int(k[0]))
            val_true = self.jointDistri(group, root, i_bin)
            val_given = k[1]
            diff.append(abs(val_true - float(val_given)))
        return diff

    def jointDistri(self,group,root,test):
        parent_states=test
        joint = 1
        for nodes in group:
            if nodes.Id in root:
                state = parent_states[nodes.Id]
                joint = joint * nodes.prob[0][int(state)]
            else:
                state = parent_states[nodes.Id]
                parents = nodes.parents
                parent_state = []
                for val in parents:
                    parent_state.append(parent_states[val])
                bin_str= ''.join(str(e) for e in parent_state)
                check=int(bin_str,2)
                joint= joint * nodes.prob[check][int(state)]
        return joint

    def get_data_joint(self, observed_var, observed_var_state, query_var):
        with open(os.path.join("Data-Part-B", "Data-Problem-8", "joint.dat"), 'r') as joint_f:
            data_joint = joint_f.readlines()

        query_var_states=[] # record all the possible states
        for i in range(0, 2**len(query_var)):
            i_bin = list(bin(int(i))[2:].zfill(len(query_var)))
            query_var_states.append(i_bin)

        prob = []
        for some_state in query_var_states: # different states in query_var
            count = 0.0
            count1 = 0.0
            for line in data_joint:
                k = line.strip().split('\t')
                i_bin = binary_transfer(int(k[0])) # this is the state of joint prob
                val_given = k[1] # this is the value of joint prob

                if [i_bin[i] for i in observed_var] == observed_var_state:
                    # at least for this line, the observed data is all right
                    # this can be modified to list covertion
                    # here
                    # here
                    # here
                    count = count + float(val_given) # observed type

                    if [i_bin[int(i)] for i in query_var] == some_state:
                       count1 = count1+float(val_given)
            prob.append(count1/count)

        return prob, query_var_states

    def get_data_dataset(self,filedata_1,filedata_1_count,observed_var,observed_var_state,query_var):

        query_var_states_1 = []
        prob = []
        for i in range(0,2**len(query_var)):
            i_bin = list(bin(int(i))[2:].zfill(len(query_var)))
            query_var_states_1.append(i_bin)

        for some_state in query_var_states_1:
            count=0.0
            count1=0.0
            for d in filedata_1:
                i_bin = binary_transfer(d)
                val_given = (filedata_1_count[d])/sum(filedata_1_count)

                if [i_bin[i] for i in observed_var] == observed_var_state:
                    count=count+float(val_given)

                    if [i_bin[int(i)] for i in query_var] == some_state:
                       count1 = count1+float(val_given)
            prob.append(count1/count)
        return prob


def count_rawnumber_dataset():
    filedata_1 = []
    filedata_1_count = []
    for i in range(0, 2 ** 12): # length 4096
        filedata_1.append(i)
        filedata_1_count.append(0)

    with open(os.path.join("Data-Part-B","Data-Problem-8","dataset.dat"),'r') as f:
        d = f.readlines()
        for i in d:
            k = i.split('\n')
            filedata_1_count[int(k[0])] = filedata_1_count[int(k[0])] + 1

    return filedata_1, filedata_1_count

def binary_transfer(number):
    # a fast version
    # return a nice list
    number_binary = (bin(number)[2:]).zfill(12)
    number_binary = list(number_binary)
    number_binary.reverse()
    return number_binary

def binary_transfer_test():
    for i in range(20):
        print(binary_transfer(i))

def debug_node(group):
    for i in group:
        print(i.Id,i.values,i.parents,i.children, i.prob)


def main():
    #["IsSummer", "HasFlu", "HasFoodPoisoning", "HasHayFever", "HasPneumonia", "HasRespiratoryProblems",
    # "HasGastricProblems", "HasRash", "Coughs", "IsFatigued", "Vomits", "HasFever"]
    The_problem_graph = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    The_problem_graph_transpose = The_problem_graph.transpose()
    filedata_1, filedata_1_count = count_rawnumber_dataset()


    def initialize_cpt():
        cpt = {}

        for i in range(0, 12):
            N = np.count_nonzero(The_problem_graph_transpose[i])
            if N == 0:
                check1 = []
                check = 0
                check1.append([check, 1 - check])
                cpt[i] = check1
            else:
                check1 = []
                for j in range(0, 2**N):
                    check = 0
                    check1.append([check, 1 - check])
                cpt[i] = check1

        return cpt

    cpt = initialize_cpt()
    GrapMdel = Bayesian_Model()
    [group, root] = GrapMdel.func(The_problem_graph, cpt)

    cpt1 = {}
    for node in group:
        if node.Id == 0:
            n = 0
            for i in filedata_1:
                i_binary = binary_transfer(i)
                if i_binary[0] == '0':
                    n = n + filedata_1_count[i]
            node.prob[0][0] = n / sum(filedata_1_count)
            node.prob[0][1] = 1 - node.prob[0][0]
            cpt1[node.Id] = ([node.prob[0][0], node.prob[0][1]])
        else:
            check1 = []
            parents_data = []
            for values in node.parents:
                parents_data.append(values)
            n = len(parents_data)
            for i in range(0, (2 ** n)):
                count = 0
                count1 = 0
                for j in filedata_1:
                    i_bin = binary_transfer(j)
                    flag = 0
                    parent_check = []
                    for k in range(n):
                        parent_check.append(i_bin[parents_data[k]])
                    d = int(''.join(str(x) for x in parent_check), 2)
                    if d == i:
                        flag = 1
                    if flag == 1:
                        count = count + filedata_1_count[j]
                    if flag == 1 and i_bin[node.Id] == '0':
                        count1 = count1 + filedata_1_count[j]
                node.prob[i][0] = count1 / count
                node.prob[i][1] = 1 - node.prob[i][0]
                check1.append([node.prob[i][0], node.prob[i][1]])
            cpt1[node.Id] = check1

    debug_node(group)
    #print(cpt1)

    diff = GrapMdel.accuracy_check(group, root)
    print("L-1")
    print(sum(diff))


    def query_and_its_time(observed_info, observed_state, query_info,filedata_1, filedata_1_count):
        start = time.time()
        prob_query = GrapMdel.get_data_dataset(filedata_1, filedata_1_count, observed_info, observed_state, query_info)
        print("From dataset")
        print(prob_query)
        end = time.time()
        print("Time cost")
        print(end - start)

        start = time.time()
        prob_query = GrapMdel.get_data_joint(observed_info, observed_state, query_info)
        print("From joint")
        print(prob_query)
        end = time.time()
        print("Time cost")
        print(end - start)

    query_and_its_time([8, 11], ['1', '1'], [1], filedata_1, filedata_1_count)
    query_and_its_time([4], ['1'], [7, 8, 9, 10, 11], filedata_1, filedata_1_count)
    query_and_its_time([0], ['1'], [10], filedata_1, filedata_1_count)

main()