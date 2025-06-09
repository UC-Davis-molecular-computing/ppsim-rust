from math import comb
from mpmath import hyp3f2, mpf, binomial, hyper, mp
from mpmath import psi as polygamma
import numpy as np
import numpy.typing as npt
from scipy.special import binom, gammaln  # , polygamma

mp.dps = 15  # set decimal places for mpmath calculations
# print(mp)


def main():
    from math import sqrt

    n = 10**8
    k = round(sqrt(n))
    c = 2
    g = 1
    trials = 10**6
    seed = 0

    rng = np.random.default_rng(seed)

    gammas_params = gammas_params_matching_hypo(n, k, c, g, 10)
    gammas_samples = sample_gammas_sum(rng, gammas_params, trials)
    hypo_samples = sample_hypo(rng, n, k, c, trials)
    print(f"gammas_samples: {gammas_samples}")
    print(f"hypo_samples: {hypo_samples}")


def sample_gammas_sum(
    rng: np.random.Generator, gammas: np.ndarray, size: int
) -> npt.NDArray[np.float64]:
    """
    Sample from a list of gamma distributions with parameters (shape, scale), given as a 2D numpy array,
    and then return their sum. Do this `size` times, and return the result as a 1D numpy array.
    """
    shapes = np.repeat(gammas[:, 0], size).reshape(gammas.shape[0], size)
    scales = np.repeat(gammas[:, 1], size).reshape(gammas.shape[0], size)
    samples = rng.gamma(shapes, scales)
    s = np.sum(samples, axis=0)
    return s


def sample_hypo(
    rng: np.random.Generator, n: int, k: int, c: int, size: int
) -> npt.NDArray[np.float64]:
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


def gammas_params_matching_hypo(
    n: int, k: int, o: int, g: int, num_gammas: int
) -> npt.NDArray[np.float64]:
    """
    Compute the parameters of `num_gammas` Gamma distributions, whose sum matches the mean and variance of a
    hypoexponential distribution summing exponentials having rates
    (reciprocals of expected values of individual exponentials)
        n choose c, n+g choose c, n+2*g choose c, ..., n+k-1 choose c.
    The parameters for the gammas are returned as a 2D ndarray representing pairs (shape, scale).

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
        shape, scale = gamma_matching_hypo(n + i * block_size, block_size, o, g)
        gammas_f.append((float(shape), float(scale)))

    if remainder > 0:
        # Handle the last block with the remainder
        # print(f'Block {num_gammas}: n={n+num_gammas*block_size}, k={remainder}')
        shape, scale = gamma_matching_hypo(n + num_gammas * block_size, remainder, o, g)
        gammas_f.append((shape, scale))

    gammas = np.array(gammas_f)

    if np.min(gammas) < 0:
        raise ValueError(
            "Shape and scale parameters must be positive, "
            "but gammas contains negative entries:\n"
            f"{gammas}"
        )

    return gammas


# XXX: many of the return types here should be `mpf` instead of `float`, but
# mpmath does not appear to support using mpf as a type,, despite calling it a "type"
# in their documentation: https://mpmath.org/doc/current/basics.html#number-types
def gamma_matching_hypo(n: int, k: int, o: int, g: int) -> tuple[float, float]:
    """
    Compute the parameters (shape, scale) of a Gamma distribution that matches the
    mean and variance of a hypoexponential distribution summing exponentials with rates
    n choose c, n+g choose c, n+2*g choose c, ..., n+k-1 choose c.
    """
    mean = mean_hypo(n, k, o, g)
    var = var_hypo(n, k, o, g)
    shape = mean**2 / var
    scale = var / mean
    return shape, scale


from functools import wraps
from typing import Callable, Any, TypeVar

# Type variable for the return type of the wrapped function
T = TypeVar("T")


def adaptive_precision(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that adaptively increases precision until convergence. Although this can
    be used to decorate any function, it should only be used if some of the arguments/
    variables in that function are mpmath types, which might benefit from increased precision.
    The function is run with increasingly larger precision contexts, and the results
    are compared for convergence, to adaptively test how much precision is required.
    It stops when maximum precision of 180 bits is reached, or two consecutive results
    are within a relative tolerance of 1e-15.

    The decorated function is responsible for handling its own argument conversions
    and working with mpmath types as needed. This decorator only manages the
    precision context and convergence testing.

    This decorator will:
    1. Calculate the function at increasing precisions: 53 (default), 60, 90, 120, 180 bits
    2. Stop early if consecutive results converge within relative tolerance of 1e-15
    3. Return the higher precision value when convergence is achieved

    The reason we choose multiples of 30 after the default 53 is that Python ints are represented
    by 30 bit chunks, and mpmath uses Python ints under the hood to represent the bits of its
    mantissa.

    Args:
        func: Function that returns a numeric result suitable for convergence testing

    Returns:
        Decorated function that adaptively computes the result with sufficient precision.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Precision sequence in bits
        precisions: list[int] = [53, 60, 90, 120, 180, 210, 240, 270, 300]

        prev_result: T | None = None

        for prec in precisions:
            # print(f"trying precision {prec=}")
            with mp.workprec(prec):
                # Calculate function value
                result: T = func(*args, **kwargs)

                # Check convergence if we have a previous result and both are numeric
                if (
                    prev_result is not None
                    and hasattr(result, "__abs__")
                    and hasattr(prev_result, "__abs__")
                    and hasattr(result, "__sub__")
                    and hasattr(prev_result, "__sub__")
                ):

                    try:
                        # Attempt convergence check for numeric types
                        abs_prev = abs(prev_result)  # type: ignore
                        if abs_prev > 0:
                            rel_diff = abs(prev_result - result) / abs_prev  # type: ignore
                            if rel_diff < 1e-15:
                                return result  # Return higher precision value
                    except (TypeError, AttributeError):
                        # If convergence check fails, continue to next precision
                        pass

                # print(f"  result at {prec=:3}: {result}")
                prev_result = result

        # If we reach here, return the final result at highest precision
        return prev_result  # type: ignore

    return wrapper


@adaptive_precision
def mean_hypo(n: int, k: int, o: int, g: int, special: bool = True) -> float:
    n = mpf(n)
    k = mpf(k)
    o = mpf(o)
    g = mpf(g)

    if o == 1:
        # need this special case unconditionally to avoid NaN calculation somewhere
        result = mean_hypo_o1(n, k, g)
    elif special:
        result = mean_hypo_general(n, k, o, g)
    else:
        result = general_mean_hypo_hypergeometric(n, k, o, g)
    # elif special and (o <= 8 or g <= 2):
    #     # result = None
    #     result = general_mean_hypo(n, k, o, g)
    #     # much faster than calling hyper
    #     if o == 2:
    #         result = mean_hypo_o2(n, k, g)
    #     elif o == 3:
    #         result = mean_hypo_o3(n, k, g)
    #     elif o == 4:
    #         result = mean_hypo_o4(n, k, g)
    #     elif o == 5:
    #         result = mean_hypo_o5(n, k, g)
    #     elif o == 6:
    #         # result = mean_hypo_o6_wolfram(n, k, g)
    #         result = mean_hypo_o6(n, k, g)
    #     elif o == 7:
    #         result = mean_hypo_o7(n, k, g)
    #     elif o == 8:
    #         result = mean_hypo_o8(n, k, g)
    #     elif g == 1:
    #         result = mean_hypo_g1(n, k, o)
    #     elif g == 2:
    #         result = mean_hypo_g2(n, k, o)
    #     assert result is not None

    # if result is None:
    #     # empirically, this is precise even with mp.dps = 15 (i.e., increasing precision beyond 15
    #     # does not change the first 15 decimals places, as it does with the faster special cases above)
    #     result = general_mean_hypo_hypergeometric(n, k, o, g)

    return float(result)


def mean_hypo_general(n: int, k: int, o: int, g: int) -> float:
    """
    Compute the mean of a hypoexponential distribution summing exponentials with rates
    n choose o, n+g choose o, n+2*g choose o, ..., n+(k-1)*g choose o.

    This is a general implementation that works for any value of o, using the pattern
    observed in mean_hypo_o1 through mean_hypo_o8. The formula computes two weighted
    sums of polygamma function calls, where the weights are binomial coefficients
    from the o'th row of Pascal's triangle with alternating signs.

    Args:
        n: Starting value for the binomial coefficient
        k: Number of exponentials to sum
        o: Order parameter (the 'o' in 'n choose o')
        g: Step size between consecutive binomial coefficients

    Returns:
        The mean of the hypoexponential distribution
    """
    assert g >= 1, "g must be at least 1"
    assert o >= 1, "o must be at least 1"

    # Compute the two weighted sums using Pascal's triangle coefficients
    sum1 = mpf(0)
    sum2 = mpf(0)

    for m in range(int(o)):
        # Binomial coefficient from Pascal's triangle with alternating sign
        sign = -1 if m % 2 == 1 else 1
        coeff = sign * binomial(o-1, m)

        # Arguments for polygamma function
        arg1 = k + (n - (o - 1 - m)) / g
        arg2 = (n - (o - 1 - m)) / g

        sum1 += coeff * polygamma(0, arg1)
        sum2 += coeff * polygamma(0, arg2)

    # print(f'mean_hypo_general {sum1=}')
    # print(f'mean_hypo_general {sum2=}')
    return o * (sum1 - sum2) / g


def mean_hypo_o1(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1/binomial(n + g*i, 1)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+g*i%2C+1%29
    assert g >= 1, "g must be at least 1"
    sum1 = polygamma(0, n / g + k)
    sum2 = polygamma(0, n / g)
    diff = sum1 - sum2
    return diff / g


def mean_hypo_o2(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 2)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+2%29
    arg11 = k + (n - 1) / g
    arg12 = k + n / g
    arg21 = (n - 1) / g
    arg22 = n / g
    sum1 = polygamma(0, arg11) - polygamma(0, arg12)
    sum2 = polygamma(0, arg21) - polygamma(0, arg22)
    # print(f'mean_hypo_o2 {sum1=}')
    # print(f'mean_hypo_o2 {sum2=}')
    return 2 * (sum1 - sum2) / g


def mean_hypo_o3(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 3)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+3%29
    return (
        3
        * (
            polygamma(0, k + (n - 2) / g)
            - 2 * polygamma(0, k + (n - 1) / g)
            + polygamma(0, k + n / g)
            - (
                polygamma(0, (n - 2) / g)
                - 2 * polygamma(0, (n - 1) / g)
                + polygamma(0, n / g)
            )
        )
    ) / g


def mean_hypo_o4(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 4)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+4%29
    return (
        4
        * (
            polygamma(0, k + (n - 3) / g)
            - 3 * polygamma(0, k + (n - 2) / g)
            + 3 * polygamma(0, k + (n - 1) / g)
            - polygamma(0, k + n / g)
            - (
                polygamma(0, (n - 3) / g)
                - 3 * polygamma(0, (n - 2) / g)
                + 3 * polygamma(0, (n - 1) / g)
                - polygamma(0, n / g)
            )
        )
    ) / g


def mean_hypo_o5(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 5)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+5%29
    assert g >= 1, "g must be at least 1"
    # print(f'mean_hypo_o5: n={n}, k={k}, g={g}')
    return (
        5
        * (
            polygamma(0, k + (n - 4) / g)
            - 4 * polygamma(0, k + (n - 3) / g)
            + 6 * polygamma(0, k + (n - 2) / g)
            - 4 * polygamma(0, k + (n - 1) / g)
            + polygamma(0, k + n / g)
            - (
                polygamma(0, (n - 4) / g)
                - 4 * polygamma(0, (n - 3) / g)
                + 6 * polygamma(0, (n - 2) / g)
                - 4 * polygamma(0, (n - 1) / g)
                + polygamma(0, n / g)
            )
        )
    ) / g


def mean_hypo_o6(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 6)
    # This is the next obvious thing: coefficients are from g'th row of Pascal's triangle.
    assert g >= 1, "g must be at least 1"
    # print(f'mean_hypo_o5: n={n}, k={k}, g={g}')
    return (
        6
        * (
            polygamma(0, k + (n - 5) / g)
            - 5 * polygamma(0, k + (n - 4) / g)
            + 10 * polygamma(0, k + (n - 3) / g)
            - 10 * polygamma(0, k + (n - 2) / g)
            + 5 * polygamma(0, k + (n - 1) / g)
            - polygamma(0, k + n / g)
            - (
                polygamma(0, (n - 5) / g)
                - 5 * polygamma(0, (n - 4) / g)
                + 10 * polygamma(0, (n - 3) / g)
                - 10 * polygamma(0, (n - 2) / g)
                + 5 * polygamma(0, (n - 1) / g)
                - polygamma(0, n / g)
            )
        )
    ) / g


def mean_hypo_o7(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 7)
    assert g >= 1, "g must be at least 1"
    return (
        7
        * (
            polygamma(0, k + (n - 6) / g)
            - 6 * polygamma(0, k + (n - 5) / g)
            + 15 * polygamma(0, k + (n - 4) / g)
            - 20 * polygamma(0, k + (n - 3) / g)
            + 15 * polygamma(0, k + (n - 2) / g)
            - 6 * polygamma(0, k + (n - 1) / g)
            + polygamma(0, k + n / g)
            - (
                polygamma(0, (n - 6) / g)
                - 6 * polygamma(0, (n - 5) / g)
                + 15 * polygamma(0, (n - 4) / g)
                - 20 * polygamma(0, (n - 3) / g)
                + 15 * polygamma(0, (n - 2) / g)
                - 6 * polygamma(0, (n - 1) / g)
                + polygamma(0, n / g)
            )
        )
    ) / g


def mean_hypo_o8(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 8)
    assert g >= 1, "g must be at least 1"
    return (
        8
        * (
            polygamma(0, k + (n - 7) / g)
            - 7 * polygamma(0, k + (n - 6) / g)
            + 21 * polygamma(0, k + (n - 5) / g)
            - 35 * polygamma(0, k + (n - 4) / g)
            + 35 * polygamma(0, k + (n - 3) / g)
            - 21 * polygamma(0, k + (n - 2) / g)
            + 7 * polygamma(0, k + (n - 1) / g)
            - polygamma(0, k + n / g)
            - (
                polygamma(0, (n - 7) / g)
                - 7 * polygamma(0, (n - 6) / g)
                + 21 * polygamma(0, (n - 5) / g)
                - 35 * polygamma(0, (n - 4) / g)
                + 35 * polygamma(0, (n - 3) / g)
                - 21 * polygamma(0, (n - 2) / g)
                + 7 * polygamma(0, (n - 1) / g)
                - polygamma(0, n / g)
            )
        )
    ) / g


def general_mean_hypo_hypergeometric(n: int, k: int, o: int, g: int) -> float:
    # mpmath does not have a special implementation of hypergeometric 4F3 or above,
    # so we use the general implementation:
    # https://mpmath.org/doc/current/functions/hypergeometric.html#hyper
    # Wolfram alpha does not have a closed form for this, but we can infer the closed
    # form from the first few formulas for g=1, g=2, g=3:
    # g=1: https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+i%2C+o%29
    # g=2: https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+2*i%2C+o%29
    # g=3: https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+3*i%2C+o%29
    # XXX: This is VERY slow, it can take over a second to compute for example, for n=10**8, k=10**4, o=4, g=4
    # So we may just want to write out the special cases for larger o
    # and raise an exception if o is larger than that.

    as1 = [1.0]
    bs1 = []
    as2 = [1.0]
    bs2 = []

    for i in range(1, int(g) + 1):
        as1.append((n - o + i) / g)
        bs1.append((n + i) / g)

        as2.append(k + (n - o + i) / g)
        bs2.append(k + (n + i) / g)

    num1 = hyper(as1, bs1, 1)
    den1 = binomial(n, o)
    num2 = hyper(as2, bs2, 1)
    den2 = binomial(n + g * k, o)

    return num1 / den1 - num2 / den2






def var_hypo(n: int, k: int, o: int, g: int, special: bool = True) -> float:
    if special:  # faster for constants c=1,2,3,4 than calling hyp3f2 for general c
        if o == 1:
            return var_hypo_o1(n, k, g)
        elif o == 2:
            return var_hypo_o2(n, k, g)
        elif o == 3:
            return var_hypo_o3(n, k, g)
        elif o == 4:
            return var_hypo_o4(n, k, g)
        elif o == 4:
            return var_hypo_o5(n, k, g)
    raise NotImplementedError


def var_hypo_o1(n: int, k: int, g: int) -> float:
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+g*i%2C+1%29%5E2
    return (polygamma(1, n/g) - polygamma(1, k + n/g)) / g**2


def var_hypo_o2(n: int, k: int, g: int) -> float:
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+g*i%2C+2%29%5E2
    return \
        (
            4 * 
            (
                2 * g * 
                (
                      polygamma(0, k + n/g) 
                    - polygamma(0, k + (n - 1)/g) 
                    + polygamma(0, (n - 1)/g) 
                    - polygamma(0, n/g) 
                )
                - 
                (
                      polygamma(1, k + n/g)
                    + polygamma(1, k + (n - 1)/g) 
                    + polygamma(1, (n - 1)/g) 
                    + polygamma(1, n/g)
                )
            )
        ) / g**2 


def var_hypo_o3(n: int, k: int, g: int) -> float:
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+g*i%2C+3%29%5E2
    return \
        (
            9 * 
            (
                3 * g * 
                (
                      polygamma(0, k + n/g) 
                    - polygamma(0, k + (n - 2)/g) 
                    + polygamma(0, (n - 2)/g) 
                    - polygamma(0, n/g)
                )
                - 
                (
                    polygamma(1, k + (g * n - 2)/g) 
                    + 4 * polygamma(1, k + (n - 1)/g) 
                    + polygamma(1, k + n/g)
                )
                + 
                (
                    polygamma(1, (n - 2)/g) 
                    + 4 * polygamma(1, (n - 1)/g) 
                    + polygamma(1, n/g)
                )
            )
        ) / g**2


def var_hypo_o4(n: int, k: int, g: int) -> float:
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+g*i%2C+4%29%5E2
    return \
        (
            16 * 
            (
                g *
                (
                    9 * 
                    (
                        - polygamma(0, (g * k + n - 2)/g) 
                        + polygamma(0, (g * k + n - 1)/g) 
                        + polygamma(0, (n - 2)/g) 
                        - polygamma(0, (n - 1)/g) 
                    )
                    - (11/3) * 
                    (
                        + polygamma(0, (g * k + n - 3)/g) 
                        - polygamma(0, k + n/g) 
                        - polygamma(0, (n - 3)/g) 
                        + polygamma(0, n/g)
                    )
                )
                - (9 * 
                    (
                        + polygamma(1, (g * k + n - 2)/g) 
                        + polygamma(1, (g * k + n - 1)/g)
                        - polygamma(1, (n - 2)/g) 
                        - polygamma(1, (n - 1)/g) 
                    )
                    +
                    (
                        polygamma(1, (g * k + n - 3)/g) 
                        + polygamma(1, k + n/g)  
                        - polygamma(1, (n - 3)/g) 
                        - polygamma(1, n/g)
                    )
                )
            )
        ) / (g**2)


def var_hypo_o5(n: int, k: int, g: int) -> float:
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+g*i%2C+5%29%5E2
    raise NotImplementedError

###################################################
## more direct ways to compute; used to verify faster ways give same answer
## these assume g = 1; rewrite for general g


def reciprocals(n: int, k: int, o: int) -> npt.NDArray[np.float64]:
    indices = np.arange(k)
    binomial_values = binom(n + indices, o)
    return 1.0 / binomial_values


def reciprocals_gamma(n: int, k: int, o: int, square: bool) -> npt.NDArray[np.float64]:
    indices = np.arange(k)
    log_binom = gammaln(n + indices + 1) - gammaln(o + 1) - gammaln(n + indices - o + 1)
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


# # I don't know what happened here, but going from o=5 to o=6 seems to be qualitatively more complex,
# # and also its slow to compute, no better than calling mpmath.hyper (which is too slow for us).
# # I don't know why Wolfram alpha did not continue with the Pascal's triangle pattern,
# # but this seems strictly worse than `mean_hypo_o6` above.
# def mean_hypo_o6_wolfram(n: int, k: int, g: int) -> float:
#     # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 6)
#     # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+6%29
#     assert g >= 1, "g must be at least 1"
#     # print(f'mean_hypo_o5_wolfram: n={n}, k={k}, g={g}')
#     return (
#         (
#             (
#                 g**6 * k**6
#                 + 3 * g**5 * k**5 * (2 * n - 5)
#                 + 5 * g**4 * k**4 * (3 * n**2 - 15 * n + 17)
#                 + 5 * g**3 * k**3 * (4 * n**3 - 30 * n**2 + 68 * n - 45)
#                 + g**2 * k**2 * (15 * n**4 - 150 * n**3 + 510 * n**2 - 675 * n + 274)
#                 + g
#                 * k
#                 * (6 * n**5 - 75 * n**4 + 340 * n**3 - 675 * n**2 + 548 * n - 120)
#                 + n * (n**5 - 15 * n**4 + 85 * n**3 - 225 * n**2 + 274 * n - 120)
#             )
#             * (
#                 polygamma(0, k + (n - 5) / g)
#                 - 5 * polygamma(0, k + (n - 4) / g)
#                 + 10 * polygamma(0, k + (n - 3) / g)
#                 - 10 * polygamma(0, k + (n - 2) / g)
#                 + 5 * polygamma(0, k + (n - 1) / g)
#                 - polygamma(0, k + n / g)
#             )
#         )
#         / binomial(g * k + n, 6)
#         - (
#             n
#             * (n**5 - 15 * n**4 + 85 * n**3 - 225 * n**2 + 274 * n - 120)
#             * (
#                 polygamma(0, (n - 5) / g)
#                 - 5 * polygamma(0, (n - 4) / g)
#                 + 10 * polygamma(0, (n - 3) / g)
#                 - 10 * polygamma(0, (n - 2) / g)
#                 + 5 * polygamma(0, (n - 1) / g)
#                 - polygamma(0, n / g)
#             )
#         )
#         / binomial(n, 6)
#     ) / (120 * g)


# def mean_hypo_g1(n: int, k: int, o: int) -> float:
#     # sum_{i=0}^{k-1} 1/binomial(n + i, o)
#     # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+i%2C+o%29
#     assert o >= 1, "o must be at least 1"
#     return (n / binomial(n, o) - (k + n) / binomial(k + n, o)) / (o - 1)


# def mean_hypo_g2(n: int, k: int, o: int) -> float:
#     # sum_{i=0}^{k-1} 1/binomial(n + 2*i, o)
#     # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+2*i%2C+o%29
#     assert o >= 1, "o must be at least 1"
#     return hyp3f2(
#         1, n / 2 - o / 2 + 1 / 2, n / 2 - o / 2 + 1, n / 2 + 1 / 2, n / 2 + 1, 1
#     ) / binomial(n, o) - hyp3f2(
#         1,
#         k + n / 2 - o / 2 + 1 / 2,
#         k + n / 2 - o / 2 + 1,
#         k + n / 2 + 1 / 2,
#         k + n / 2 + 1,
#         1,
#     ) / binomial(
#         2 * k + n, o
#     )



if __name__ == "__main__":
    main()
