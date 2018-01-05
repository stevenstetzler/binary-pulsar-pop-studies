from scipy import stats
from scipy.special import gamma
from scipy.stats import expon
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
from matplotlib import pyplot as plt
import corner

def get_measurements(alpha, beta, N=30, noise_draws=100, plot=False):
    dist = stats.beta(alpha, beta)
    measurements = np.zeros((N, noise_draws))
    for i in range(N):
        cosi = dist.rvs(1)[0]
        noise = stats.norm(cosi, 0.05).rvs(noise_draws)
        # only include draws in [0, 1]
        noise[noise < 0] = 0
        noise[noise > 1] = 1
        measurements[i] = noise
    if plot:
        plt.hist(np.median(measurements, axis=1), normed=True, label="Data")
        plt.plot(np.linspace(0, 1, 1000), dist.pdf(np.linspace(0, 1, 1000)), label="Population")
        plt.legend()
    return measurements

def get_samples(alpha, beta, measurements):
    guess = stats.beta(alpha, beta)
    # Draw new samples from P(COSI_i | alpha, beta)
    N = len(measurements)
    samples = np.zeros(N)
    for i in range(N):
        probs = guess.pdf(measurements[i])
        probs /= np.sum(probs)
        samples[i] = np.random.choice(measurements[i], p=probs)
    return samples

def get_alpha(alpha, beta, samples):
    
    def _lgprior(alpha, beta):
        return math.log(1)
    
    def _lglike(samples, alpha, beta):
        return len(samples) * (math.log(gamma(alpha + beta)) - math.log(gamma(alpha))) + np.sum(alpha * np.log(samples))

    # log-posterior for P(alpha | beta, samples)
    def _lgprob(alpha, beta, samples):
        if alpha + beta < 0:
            return -np.inf
        if alpha < 0:
            return -np.inf
        if alpha + beta > 20:
            return -np.inf
        if alpha > 20:
            return -np.inf
        
        # l(alpha | beta, samples) ~ l(alpha, beta) + l(samples | alpha, beta)
        return _lgprior(alpha, beta) + _lglike(samples, alpha, beta)

    # Draw new alpha from proposal distribution
    prop = expon(scale=1/0.25)
    
    new_alpha = prop.rvs(1)[0]
    
    prob_new_alpha = prop.pdf(new_alpha)
    prob_old_alpha = prop.pdf(alpha)
    
    # MH-Step
    # a = a_1 * a_2
    # a_1 = P(new) / P(old)
    # a_2 = Q(old) / Q(new)
    # Accept with prob a if a <= 1
    # lg(a) = lg(a_1) + lg(a2)
    # lg(a) = lg(P(new)) - lg(P(old)) + lg(Q(old)) - lg(Q(new))
    # Accept with prob a if lg(a) <= 0
    
    try:
        lg_a1 = _lgprob(new_alpha, beta, samples) - _lgprob(alpha, beta, samples)
    except:
        print "bad lg_a1 alpha"
    
    try:
        lg_a2 = math.log(prob_old_alpha) - math.log(prob_new_alpha)
    except:
        print "bad lg_a2 alpha"
    
    lg_a = lg_a1 + lg_a2
    
    a = math.exp(lg_a)
    
    if a > 1:
        return new_alpha
    else:
        r = np.random.uniform()
        if r < a:
            return new_alpha
        else:
            return alpha

def get_beta(alpha, beta, samples):
    
    def _lgprior(alpha, beta):
        return math.log(1)
    
    def _lglike(samples, alpha, beta):
        return len(samples) * (math.log(gamma(alpha + beta)) - math.log(gamma(beta))) + np.sum(beta * np.log(1 - samples))

    # log-posterior for P(alpha | beta, samples)
    def _lgprob(alpha, beta, samples):
        if alpha + beta < 0:
            return -np.inf
        if alpha < 0:
            return -np.inf
        if alpha + beta > 20:
            return -np.inf
        if alpha > 20:
            return -np.inf
        
        # l(beta | alpha, samples) ~ l(alpha, beta) + l(samples | alpha, beta)
        return _lgprior(alpha, beta) + _lglike(samples, alpha, beta)
    
    # Draw new beta from proposal distribution
    prop = expon(scale=1/0.25)
    
    new_beta = prop.rvs(1)[0]
    
    prob_new_beta = prop.pdf(new_beta)
    prob_old_beta = prop.pdf(beta)
    
    # MH-Step
    # a = a_1 * a_2
    # a_1 = P(new) / P(old)
    # a_2 = Q(old) / Q(new)
    # Accept with prob a if a <= 1
    # lg(a) = lg(a_1) + lg(a2)
    # lg(a) = lg(P(new)) - lg(P(old)) + lg(Q(old)) - lg(Q(new))
    # Accept with prob a if lg(a) <= 0
    
    try:
        lg_a1 = _lgprob(alpha, new_beta, samples) - _lgprob(alpha, beta, samples)
    except:
        print "bad lg_a1 beta"
        
    try:
        lg_a2 = math.log(prob_old_beta) - math.log(prob_new_beta)
    except:
        print "bad lg_a2 beta"
        
#     print lg_a1
#     print lg_a2
#     print
    
    lg_a = lg_a1 + lg_a2
    
    a = math.exp(lg_a)
    
    if a > 1:
        return new_beta
    else:
        r = np.random.uniform()
        if r < a:
            return new_beta
        else:
            return beta

def run_simulation(pop_alpha, pop_beta, N, burn_in=0.5, thin=0.1, guess_alpha=1, guess_beta=1, niter=1000):
    measurements = get_measurements(pop_alpha, pop_beta, N=N)
    
    plt.hist(np.median(measurements, axis=1), normed=True, label="Data")
    plt.plot(np.linspace(0, 1, 100), stats.beta(pop_alpha, pop_beta).pdf(np.linspace(0, 1, 100)), label="Population")
    plt.plot(np.linspace(0, 1, 100), stats.beta(guess_alpha, guess_beta).pdf(np.linspace(0, 1, 100)), label="Guess")
    plt.legend()
    plt.show()
    
    alphas = np.zeros(niter)
    betas = np.zeros(niter)
    
    alphas[0] = guess_alpha
    betas[0] = guess_beta

    # Gibbs Sampling
    for i in range(1, niter):
        if i % 100 == 0:
            print "{0:0.1f}% Done                   \r".format(100. * i / float(niter)),
        # Draw x_i ~ P(x_i | alpha_i-1, beta_i-1)
        samples = get_samples(alphas[i - 1], betas[i - 1], measurements)
        # Draw alpha_i ~ P(alpha_i | x_i, beta_i-1)
        alphas[i] = get_alpha(alphas[i - 1], betas[i - 1], samples)
        # Draw beta_i ~ P(beta_i | x_i, alpha_i)
        betas[i] = get_beta(alphas[i], betas[i - 1], samples)
    
    alphas_good = alphas[int(burn_in * len(alphas))::int(1/thin)]
    betas_good = betas[int(burn_in * len(betas))::int(1/thin)]

    best_guess_alpha = np.median(alphas_good)
    best_guess_beta = np.median(betas_good)

    plt.hist(np.median(measurements, axis=1), normed=True, label="Data")
    plt.plot(np.linspace(0, 1, 100), stats.beta(best_guess_alpha, best_guess_beta).pdf(np.linspace(0, 1, 100)), label="Guess")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    alpha_beta_corner = np.vstack((alphas_good, betas_good)).T
    try:
        figure = corner.corner(alpha_beta_corner)
    except:
        print "Cannot corner plot"

    return alphas, betas

def main():
    return None


if __name__ == '__main__':
    main()

