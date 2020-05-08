import os
import scipy.io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = os.path.join("Data-HWK3-2020","Problem-4","SymptomDisease.mat")
data=scipy.io.loadmat(data_path)
#print(data.keys())

W = data['W']
b = data['b']
p = data['p']
s = data['s'] # symptoms

def check_data():
    print(W.shape,b.shape,p.shape,s.shape)
    # (200, 50) (200, 1) (50, 1) (200, 1)

def sigma(x):
    return 1/(1+np.exp(-x))

def p_s1_under_d(W,b,d):
    return sigma(np.dot(W,d)+b.reshape(200))

def joint(d,W,b,s,p):
    p_s1_under_d_x = p_s1_under_d(W,b,d)
    joint = np.prod((p.T**d)*(1-p.T)**(1-d))* \
            np.prod((p_s1_under_d_x**s.T)*(1-p_s1_under_d_x)**(1-s.T))
    return joint

def test_func():
    d = np.zeros(50)
    p_s1_under_d_x = p_s1_under_d(W,b,d)
    print(p_s1_under_d_x)
    print(p_s1_under_d_x.shape) # (200,)
    joint_x = joint(d,W,b,s,p)
    print(joint_x)
    print(joint_x.shape)

def main():

    total_N = 2500
    d = np.zeros(50)
    prob_check_old = p
    d_sampled = np.zeros((total_N, 50))
    a = [0] * total_N
    m = 0

    results_record = []

    for n in tqdm(range(total_N)):
        Prob_check_current = np.zeros(len(d))
        for i in range(len(d)):
            d[i]=0
            d_0= joint(d,W,b,s,p)
            d[i]=1
            d_1= joint(d,W,b,s,p)
            prob1 = d_1/(d_0+d_1)
            Prob_check_current[i]=prob1
            d[i]= np.random.binomial(1,prob1)

        if n >= 2000: # burn-in
            if n % 20 == 0: # sub-sampling
                d_sampled[m] = d
                a[m] = np.sum(np.abs(prob_check_old - Prob_check_current))
                prob_check_old = Prob_check_current
                m = m+1
                #print(Prob_check_current)
                results_record.append(Prob_check_current)

    #plt.plot(a)
    #plt.show()
    with np.printoptions(precision=3):
        print(Prob_check_current)
        print(np.mean(np.array(results_record), axis=0))

if __name__ == '__main__':
    main()


#print(p)
