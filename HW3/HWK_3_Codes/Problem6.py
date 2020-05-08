import numpy as np
from scipy.stats import norm
from numpy.linalg import cholesky
import matplotlib.pyplot as plt


############# problem a #################

def problem_a():
    x = 0.5 * (np.random.normal(-5, 1, 100)) + 0.5 * (np.random.normal(5, 1, 100))
    return x

def generate_samples(s_number=100):
    x_samples = []
    for i in range(s_number):
        u = np.random.uniform(0, 1)
        if u >= 0.5:
            x_samples.append(np.random.normal(5.0, 1))
        else:
            x_samples.append(np.random.normal(-5.0, 1))
    return np.array(x_samples)

#s = generate_samples()
#print(s)
#print(s.shape)

############# problem b #################

def prior(mu1, mu2):
    # Gaussian prior
    return -(mu1**2+mu2**2)/200

def posterior_p(x_samples, mu1, mu2):
    # posterior
    p_x_under_mu = 0.5*np.exp(-0.5*(x_samples - mu1)**2) + 0.5*np.exp(-0.5*(x_samples - mu2)**2)
    p_x_under_mu = np.log(p_x_under_mu)
    p_x_under_mu = np.sum(p_x_under_mu)
    return p_x_under_mu + prior(mu1, mu2)

def accept_rate(x_samples, mu1, mu2, mu1_cap, mu2_cap):
    ar = posterior_p(x_samples,mu1_cap,mu2_cap) - posterior_p(x_samples,mu1,mu2)
    return np.exp(ar)

def MH_sampling(x_samples, sigma):
    #initial_mu1, initial_mu2 = 0.0, 0.0
    #mu1 = initial_mu1
    #mu2 = initial_mu2
    mu1 = 0
    mu2 = 0
    mu1_record = []
    mu2_record = []

    N = 11000 # discard the first 10000 samples and plot the next 1000 samples
    accept = 0

    for n in range(N):
        mu1_cap, mu2_cap = np.random.multivariate_normal([mu1, mu2], [[sigma**2, 0], [0, sigma**2]], 1)[0]
        ar = accept_rate(x_samples, mu1, mu2, mu1_cap, mu2_cap)

        if ar >= 1:
            mu1, mu2 = mu1_cap, mu2_cap
            accept += 1
        else:
            tmp = np.random.uniform(0, 1)
            if tmp <= ar:
                mu1, mu2 = mu1_cap, mu2_cap
                accept += 1
        if n >= 10000:
            mu1_record.append(mu1)
            mu2_record.append(mu2)

    plt.plot(mu1_record, mu2_record, '+')
    plt.title("MH_sigma{}".format(sigma))
    plt.show()

    return np.mean(mu1_record), np.mean(mu2_record), accept/N

############# problem c #################

def Gibbs_sampling(x_samples):
    mu1 = 0
    mu2 = 0
    N = 11000

    mu1_record = []
    mu2_record = []


    for n in range(N):
        prob_record = []
        for i in range(100):
            prob = np.exp(-0.5 * (x_samples[i] - mu1) ** 2) / (np.exp(-0.5*(x_samples[i]-mu2)**2) + np.exp(-0.5*(x_samples[i]-mu1)**2))
            prob_record.append(prob)
        latent = np.random.binomial(1, prob_record, 100)

        st_1 = np.sum((latent == 1))
        st_0 = np.sum((latent == 0))
        mu1 = np.random.normal(100 * np.sum((latent == 1)*x_samples)/(100*st_1+1), np.sqrt(200/(100*st_1+1)))
        mu2 = np.random.normal(100 * np.sum((latent == 0)*x_samples)/(100*st_0+1), np.sqrt(200/(100*st_0+1)))

        if n >= 10000:
            mu1_record.append(mu1)
            mu2_record.append(mu2)

    plt.plot(mu1_record, mu2_record, '+')
    plt.title("Gibbs")
    plt.show()

    return np.mean(mu1_record), np.mean(mu2_record)

if __name__ == "__main__":

    for i in range(6):
        x_samples = generate_samples()
        mu1_MH, mu2_MH, acc_rate = MH_sampling(x_samples, sigma=0.5)
        print("sigma=0.5", "ave_mu1: ", mu1_MH, "-- ave_mu2: ", mu2_MH, "-- accept_rate: ",acc_rate)

        mu1_MH, mu2_MH, acc_rate = MH_sampling(x_samples, sigma=5)
        print("sigma=5", "ave_mu1: ", mu1_MH, "-- ave_mu2: ", mu2_MH, "-- accept_rate: ", acc_rate)

        mu1_G, mu2_G = Gibbs_sampling(x_samples)
        print("ave_mu1_G: ", mu1_G, "-- ave_mu2_G: ", mu2_G)


    #plt.plot(MU1_G, MU2_G, '+')
    #plt.show()
