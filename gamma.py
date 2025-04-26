from math import comb
from mpmath import hyp3f2, mpf, mp, binomial, psi
import numpy as np
from scipy.special import binom, gammaln, polygamma

def main():
    from math import sqrt
    n = 10 ** 8
    k = round(sqrt(n))
    c = 2
    trials = 10 ** 6
    seed = 0

    rng = np.random.default_rng(seed)

    gammas = gammas_matching_hypo(n, k, c, 10)
    gammas_samples = sample_gammas_sum(rng, gammas, trials)
    hypo_samples = sample_hypo(rng, n, k, c, trials)
    print(f'gammas_samples: {gammas_samples}')
    print(f'hypo_samples: {hypo_samples}')


def sample_gammas_sum(rng: np.random.Generator, gammas: np.ndarray, size: int) -> np.ndarray:
    """
    Sample from a list of gamma distributions with parameters (shape, scale), given as a 2D numpy array,
    and then return their sum. Do this `size` times, and return the result as a 1D numpy array.
    """
    shapes = np.repeat(gammas[:, 0], size).reshape(gammas.shape[0], size)
    scales = np.repeat(gammas[:, 1], size).reshape(gammas.shape[0], size)
    # print(f'shapes: {shapes}')
    # print(f'scales: {scales}')
    samples = rng.gamma(shapes, scales)
    s = np.sum(samples, axis=0)
    # print(f'samples: {samples}')
    # print(f's: {s}')
    return s


def sample_hypo(rng: np.random.Generator, n: int, k: int, c: int, size: int) -> np.ndarray:
    """
    Sample from a hypoexponential distribution summing exponentials having rates
    n choose c, n+1 choose c, n+2 choose c, ..., n+k-1 choose c.
    (directly, by sampling k exponentials with those rates and summing them)
    """
    indices = np.arange(k)
    scales = 1.0 / binom(n + indices, c)
    scales = np.repeat(scales, size).reshape(scales.shape[0], size)
    exp_samples = rng.exponential(scales, size=(k, size))
    samples = np.sum(exp_samples, axis=0)
    return samples


def gammas_matching_hypo(n: int, k: int, c: int, num_gammas: int) -> np.ndarray:
    """
    Compute the parameters of `num_gammas` Gamma distributions, whose sum matches the mean and variance of a
    hypoexponential distribution summing exponentials having scales (expected values of individual
    exponentials) n choose c, n+1 choose c, n+2 choose c, ..., n+k-1 choose c.
    The parameters are returned as a list of tuples (shape, scale).

    If `num_gammas` evenly divides `k`, so that `k` / `num_gammas` is a integer `s`, each gamma distribution
    is chosen to match a hypoexponential distribution with `s` exponentials. The i'th such
    gamma distribution has the same mean and variance as the hypoexponential distribution
    corresponding to the i'th block of `s` exponentials in the original hypoexponential distribution.
    If `num_gammas` does not evenly divide `k`, the last gamma distribution is chosen to match a
    hypoexponential distribution corresponding to the final `k` % `num_gammas` exponentials in the
    original hypoexponential.
    """
    if num_gammas > k:
        raise ValueError("num_gammas must be less than or equal to k")
    if num_gammas <= 0:
        raise ValueError("num_gammas must be greater than 0")

    # Calculate the number of exponentials in each block
    block_size = k // num_gammas
    remainder = k % num_gammas

    gammas_f: list[tuple[float, float]] = []
    for i in range(num_gammas):
        # print(f'Block {i}: n={n+i*block_size}, k={block_size}')
        shape, scale = gamma_matching_hypo(n + i * block_size, block_size, c)
        gammas_f.append((float(shape), float(scale)))

    if remainder > 0:
        # Handle the last block with the remainder
        # print(f'Block {num_gammas}: n={n+num_gammas*block_size}, k={remainder}')
        shape, scale = gamma_matching_hypo(n + num_gammas * block_size, remainder, c)
        gammas_f.append((shape, scale))

    gammas: np.ndarray = np.array(gammas_f)

    if np.min(gammas) < 0:
        raise ValueError("Shape and scale parameters must be positive, "
                         "but gammas contains negative entries:\n"
                         f"{gammas}")

    return gammas


def gamma_matching_hypo(n: int, k: int, c: int) -> tuple[mpf, mpf]:
    """
    Compute the parameters of an Gamma distribution that matches the mean and variance of a hypoexponential
    distribution summing exponentials having rates
    n choose c, n+1 choose c, n+2 choose c, ..., n+k-1 choose c.
    """
    mean = mean_hypo(n, k, c)
    var = var_hypo(n, k, c)
    shape = mean ** 2 / var
    scale = var / mean
    return shape, scale


def mean_hypo(n: int, k: int, c: int, special: bool = True) -> mpf:
    if c == 1:  # need this special case unconditionally to avoid div by 0
        return mean_hypo1(n, k)
    if special:  # slightly faster for c=2,3; not a huge difference
        if c == 2:
            return mean_hypo2(n, k)
        elif c == 3:
            return mean_hypo3(n, k)
    # return (n / comb(n, c) - (n + k) / comb(n + k, c)) / (c - 1)
    n = mpf(n)
    k = mpf(k)
    c = mpf(c)
    return (n / binomial(n, c) - (n + k) / binomial(n + k, c)) / (c - 1)


def var_hypo(n: int, k: int, c: int, special: bool = True) -> mpf:
    if special:  # faster for constants c=1,2,3,4 than calling hyp3f2 for general c
        if c == 1:
            return var_hypo1(n, k)
        elif c == 2:
            return var_hypo2(n, k)
        elif c == 3:
            return var_hypo3(n, k)
        elif c == 4:
            return var_hypo4(n, k)
    n_choose_c = binomial(n, c)
    n_plus_k_choose_c = binomial(n + k, c)
    return hyp3f2(1, n - c + 1, n - c + 1, n + 1, n + 1, 1) / n_choose_c ** 2 \
        - hyp3f2(1, n + k - c + 1, n + k - c + 1, n + k + 1, n + k + 1, 1) / n_plus_k_choose_c ** 2


def mean_hypo1(n: int, k: int) -> mpf:
    # return float(polygamma(0, n + k) - polygamma(0, n))
    return psi(0, n + k) - psi(0, n)


def mean_hypo2(n: int, k: int) -> mpf:
    n = mpf(n)
    k = mpf(k)
    return 2 * k / ((n + k - 1) * (n - 1))


def mean_hypo3(n: int, k: int) -> mpf:
    n = mpf(n)
    k = mpf(k)
    return (3 * k * (2 * n + k - 3)) / ((n - 2) * (n - 1) * (n + k - 2) * (n + k - 1))


def var_hypo1(n: int, k: int) -> mpf:
    # return float(polygamma(1, n) - polygamma(1, n + k))
    return psi(1, n) - psi(1, n + k)


def var_hypo2(n: int, k: int) -> mpf:
    # return mpf(4) * k * (k - 2 * k * n - 2 * n * (n - 1)) / ((n - 1) * (n + k - 1)) ** 2 + \
    #     8 * (polygamma(1, n - 1) - polygamma(1, n + k - 1))
    n = mpf(n)
    k = mpf(k)
    return mpf(4) * k * (k - 2 * k * n - 2 * n * (n - 1)) / ((n - 1) * (n + k - 1)) ** 2 + \
        8 * (psi(1, n - 1) - psi(1, n + k - 1))


def var_hypo3(n: int, k: int) -> mpf:
    n = mpf(n)
    k = mpf(k)
    return (mpf(9) * (6 * (n ** 2 - 3 * n + 2) ** 2 * (
            k ** 2 + k * (2 * n - 3) + n ** 2 - 3 * n + 2) ** 2 * psi(1, n - 2) - 6 * (
                         n ** 2 - 3 * n + 2) ** 2 * (
                         k ** 2 + k * (2 * n - 3) + n ** 2 - 3 * n + 2) ** 2 * psi(1,
                                                                                         k + n - 2) - k * (
                         k ** 3 * (6 * n ** 3 - 21 * n ** 2 + 25 * n - 9) + 2 * k ** 2 * (
                         9 * n ** 4 - 42 * n ** 3 + 74 * n ** 2 - 57 * n + 15) + k * (
                                 18 * n ** 5 - 105 * n ** 4 + 246 * n ** 3 - 288 * n ** 2 + 163 * n - 33) + 6 * n ** 6 - 42 * n ** 5 + 123 * n ** 4 - 192 * n ** 3 + 163 * n ** 2 - 66 * n + 8))) / (
            (n - 2) ** 2 * (n - 1) ** 2 * (k + n - 2) ** 2 * (k + n - 1) ** 2)


def var_hypo4(n: int, k: int) -> mpf:
    # TODO: precompute shared subexpressions here (e.g., n**3, n**2, (n ** 3 - 6 * n ** 2 + 11 * n - 6))
    n = mpf(n)
    k = mpf(k)
    numerator = (
            mpf(32) * (
            30 * (n ** 3 - 6 * n ** 2 + 11 * n - 6) ** 2 *
            (k ** 3 + 3 * k ** 2 * (n - 2) + k * (
                    3 * n ** 2 - 12 * n + 11) + n ** 3 - 6 * n ** 2 + 11 * n - 6) ** 2 *
            psi(1, n - 3) -

            30 * (n ** 3 - 6 * n ** 2 + 11 * n - 6) ** 2 *
            (k ** 3 + 3 * k ** 2 * (n - 2) + k * (
                    3 * n ** 2 - 12 * n + 11) + n ** 3 - 6 * n ** 2 + 11 * n - 6) ** 2 *
            psi(1, k + n - 3) -

            k * (
                    k ** 5 * (30 * n ** 5 - 255 * n ** 4 + 845 * n ** 3 - 1350 * n ** 2 + 1039 * n - 303) +
                    6 * k ** 4 * (
                            25 * n ** 6 - 255 * n ** 5 + 1065 * n ** 4 - 2320 * n ** 3 + 2774 * n ** 2 - 1721 * n + 426) +
                    k ** 3 * (
                            300 * n ** 7 - 3570 * n ** 6 + 17955 * n ** 5 - 49350 * n ** 4 + 79925 * n ** 3 - 76170 * n ** 2 + 39382 * n - 8394) +
                    k ** 2 * (
                            300 * n ** 8 - 4080 * n ** 7 + 23975 * n ** 6 - 79380 * n ** 5 + 161750 * n ** 4 - 207480 * n ** 3 + 163283 * n ** 2 - 71652 * n + 13212) +
                    k * (
                            150 * n ** 9 - 2295 * n ** 8 + 15420 * n ** 7 - 59640 * n ** 6 + 146145 * n ** 5 - 235050 * n ** 4 + 247802 * n ** 3 - 164592 * n ** 2 + 61963 * n - 9879) +
                    30 * n ** 10 - 510 * n ** 9 + 3855 * n ** 8 - 17040 * n ** 7 + 48715 * n ** 6 - 94020 * n ** 5 + 123901 * n ** 4 - 109728 * n ** 3 + 61963 * n ** 2 - 19758 * n + 2592
            )
    )
    )

    denominator = (
            mpf(3) * (n - 3) ** 2 * (n - 2) ** 2 * (n - 1) ** 2 * (k + n - 3) ** 2 * (k + n - 2) ** 2 * (
            k + n - 1) ** 2
    )

    return numerator / denominator


###################################################
## more direct ways to compute; used to verify faster ways give same answer


def reciprocals(n: int, k: int, c: int) -> np.ndarray:
    indices = np.arange(k)
    binomial_values = binom(n + indices, c)
    return 1.0 / binomial_values


def reciprocals_gamma(n: int, k: int, c: int, square: bool) -> np.ndarray:
    indices = np.arange(k)
    log_binom = (gammaln(n + indices + 1) - gammaln(c + 1) -
                 gammaln(n + indices - c + 1))
    coef = -2 if square else -1
    return np.exp(coef * log_binom)


def mean_direct_np_gamma(n: int, k: int, c: int) -> float:
    return np.sum(reciprocals_gamma(n, k, c, False))


def var_direct_np_gamma(n: int, k: int, c: int) -> float:
    return np.sum(reciprocals_gamma(n, k, c, True))


def mean_direct_np(n: int, k: int, c: int) -> float:
    return np.sum(reciprocals(n, k, c))


def var_direct_np(n: int, k: int, c: int) -> float:
    return np.sum(reciprocals(n, k, c) ** 2)  # type: ignore


def mean_direct(n: int, k: int, c: int) -> float:
    s = 0
    for i in range(k):
        s += 1 / comb(n + i, c)
    return s


def var_direct(n: int, k: int, c: int) -> float:
    s = 0
    for i in range(k):
        s += 1 / comb(n + i, c) ** 2
    return s


if __name__ == '__main__':
    main()
