#! /usr/bin/env python
# math2.py
"""
math2.py

Written for Python 2.5-2.7.

This library is an extension of the standard math module. As it extends
math and not cmath, all functions here do not work with complex numbers.

In addition to providing all the functions in the math module, it also
contains implementations of many common math functions not found in the
math module.

However, note that with the release of later versions of Python, some of
these functions are now included in the math module. In the case that a
function is duplicated under the version of Python this module runs on,
the function here will simply serve as a copy of the math module version.
"""

import sys as _sys
import random as _random
from math import *

_MAJOR_VERSION_N, _MINOR_VERSION_N = _sys.version_info[:2]

######################################################################
## helper functions
######################################################################

def _quickselect(array, k):
    """
    Simple implementation of quickselect.
    """
    if len(array) < 2:
        return array[k]

    pivot = _random.randrange(len(array))
    # partition
    array[0], array[pivot] = array[pivot], array[0]
    i = 1
    j = len(array) - 1
    while True:
        while i < len(array) and array[i] <= array[0]:
            i += 1
        while j >= 1 and array[j] > array[0]:
            j -= 1

        if j < i:
            break

        array[i], array[j] = array[j], array[i]

    array[0], array[j] = array[j], array[0]

    # select
    if j == k:
        return array[j]
    elif j > k:
        return _quickselect(array[:j], k)
    elif j < k:
        return _quickselect(array[j+1:], k-j-1)

######################################################################
## series and summations
######################################################################

if _MINOR_VERSION_N < 6:
    def fsum(iterable, start=0):
        """
        Sums the elements in the given iterable with more precision than
        a naive, straightforward summation.

        This uses the Kahan summation technique.

        Equivalent to math.fsum in concept, but that isn't available until
        Python 2.6. However, this adds support for a start argument.
        """
        total = start
        c = 0
        for item in iterable:
            y = item - c
            t = total + y
            c = (t - total) - y
            total = t
        return total
else:
    # add on a start argument to math.fsum
    _old_fsum = fsum
    def fsum(iterable, start=0):
        return _old_fsum([start, _old_fsum(iterable)])

def arithmetic_sum(n, a=1, d=1):
    """
    Returns the sum of the arithmetic progression

    a + (a + d) + (a + 2d) + ... + (a + (n-1)d)

    By default, a=1 and d=1, to handle the common case

    1 + 2 + 3 + 4 + ... + n

    This can technically also be done with (assuming d != 0)

    sum(xrange(a, a + (n-1)*d + (d/abs(d) * 1), d))

    but the formula is O(1), whereas the above method is O(n).
    """
    assert n >= 1, "number of elements to sum must be at least 1"
    return n * float(2*a + d*(n-1)) / 2

def geometric_sum(n, a, r):
    """
    Returns the sum of the geometric progression

    a + ar + ar^2 + ar^3 + ... + ar^(n-1)
    """
    assert n >= 1, "number of elements to sum must be at least 1"
    return float(r**n - 1) / (r-1) * a

def infinite_geometric_sum(a, r):
    """
    Returns the sum of the infinite geometric series

    a + ar + ar^2 + ar^3 + ...

    assuming |r| < 1.
    """
    assert abs(r) < 1, "no infinite geometric sum exists for |r| >= 1"
    return a / float(1 - r)

######################################################################
## number theory
######################################################################

def gcd(a, b, *args):
    """
    Finds the greatest common divisor of all numbers given (min. 2).
    """
    if not args:
        while b != 0:
            a, b = b, a % b
        return abs(a)
    else:
        return gcd(a, gcd(b, args[0], *args[1:]))

def lcm(a, b, *args):
    """
    Finds the lowest common multiple of all numbers given (min. 2).
    """
    if not args:
        if a == b == 0:
            return 0
        else:
            return abs(max(a, b)) // gcd(a, b) * abs(min(a, b))
    else:
        return lcm(a, lcm(b, args[0], *args[1:]))

######################################################################
## statistics - center of data
######################################################################

def arithmetic_mean(a, *args):
    """
    Computes the arithmetic mean of the given arguments (min. 1).
    """
    return fsum(args, a) / float(len(args) + 1)

# handy synonyms
average = mean = arithmetic_mean

def geometric_mean(a, *args):
    """
    Computes the geometric mean of the given arguments (min. 1).
    
    All arguments must be positive.
    """
    assert a > 0, "arguments to geometric_mean must be positive"
    
    p = a
    for arg in args:
        assert arg > 0, "arguments to geometric_mean must be positive"
        p *= arg
    return pow(p, 1.0/(len(args)+1))

def harmonic_mean(a, *args):
    """
    Computes the harmonic mean of the given arguments (min. 1).

    All arguments must be positive.
    """
    assert a > 0, "arguments to harmonic_mean must be positive"
    for arg in args:
        assert arg > 0, "arguments to harmonic_mean must be positive"
    
    return (len(args) + 1) / fsum([(1.0/arg) for arg in args], 1.0/a)

def median(a, *args):
    """
    Computes the median of the given arguments (min. 1).
    """
    augmented_args = (a,) + args
    n = len(augmented_args)
    if n % 2 != 0:
        return _quickselect(list(augmented_args), n // 2)
    else:
        return float(_quickselect(list(augmented_args), n // 2) +
                _quickselect(list(augmented_args), n // 2 + 1)) / 2

def mode(a, *args):
    """
    Computes the mode of the given arguments (min. 1).
    
    If there is more than one "most frequent" value,
    only the first one is returned.
    """
    m = a
    augmented_args = (a,) + args
    for arg in augmented_args:
        if augmented_args.count(arg) > augmented_args.count(m):
            m = arg
    
    return m

######################################################################
## statistics - probability
######################################################################

if _MINOR_VERSION_N < 6:
    def factorial(n):
        """
        Computes n!.

        Equivalent to math.factorial, but that isn't available until Python 2.6.
        """
        if n < 0 or not isinstance(n, int):
            raise ValueError("%s not a valid argument to factorial" % n)
        f = 1
        for i in xrange(2, n+1):
            f *= i
        return f

def P(n, r):
    """
    Computes nPr.
    """
    if r > n:
        return 0
    p = n
    for i in xrange(n-1, n-r, -1):
        p *= i
    return p

def C(n, r):
    """
    Computes nCr.
    """
    return P(n, r) // factorial(r)

######################################################################
## statistics - dispersion of data
######################################################################

def pop_variance(a, *args):
    """
    Computes the population variance of the given arguments (min. 1).
    """
    augmented_args = [a] + list(args)
    m = arithmetic_mean(augmented_args)
    v = 0
    c = 0
    for arg in augmented_args:
        dist = arg - m
        y = pow(dist, 2) - c
        t = v + y
        c = (t - v) - y
        v = t
    return float(v) / (len(args) + 1)

def sample_variance(a, *args):
    """
    Computes the sample variance of the given arguments (min. 1).
    """
    if len(args) == 0:
        return 0
    else:
        return pop_variance(a, *args) * (len(args) + 1) / len(args)

def pop_standard_dev(a, *args):
    """
    Computes the population standard deviation of the given arguments
    (min. 1).
    """
    return sqrt(pop_variance(a, *args))

def sample_standard_dev(a, *args):
    """
    Computes the sample standard deviation of the given arguments
    (min. 1).
    """
    return sqrt(sample_variance(a, *args))

######################################################################
## polynomials
######################################################################

def quadratic(a, b, c):
    """
    Returns the real solutions to ax^2 + bx + c = 0.
    
    a is assumed not to be 0.
    
    The return value is always a two-element tuple. If the two solutions
    are identical, the elements in the tuple will be identical. If the
    two solutions are not real, None will be used for each solution.
    """
    if a == 0:
        raise ValueError("a cannot be zero")
    
    disc = b**2 - (4*a*c)
    if disc < 0:
        return (None, None)
    
    if b != 0:
        q = (-0.5) * (b + sgn(b) * sqrt(disc))
        x1 = q / a
        x2 = c / q
    else:
        x1 = sqrt(-c / a)
        x2 = -x1
    return x1, x2

def cubic(a, b, c, d):
    """
    Returns the real solutions to ax^3 + bx^2 + cx + d = 0.

    a is assumed not to be 0.

    The return value is always a three-element tuple. If any of the
    solutions are identical, there will be duplicates in the tuple.
    If two of the solutions are not real, None will be used for those
    solutions.
    """
    if a == 0:
        raise ValueError("a cannot be zero")

    # need to derive efficient numeric solution of cubics
    # (that are guaranteed to work)
    raise NotImplemented

######################################################################
## trigonometric and hyperbolic functions
######################################################################

def csc(x):
    """
    Trigonometric function cosecant (reciprocal of sine).
    """
    return 1 / sin(x)

def sec(x):
    """
    Trigonometric function secant (reciprocal of cosine).
    """
    return 1 / cos(x)

def cot(x):
    """
    Trigonometric function cotangent (reciprocal of tangent).
    """
    return 1 / tan(x)

def acsc(x):
    """
    Inverse of cosecant.
    """
    return asin(1 / x)

def asec(x):
    """
    Inverse of secant.
    """
    return acos(1 / x)

def acot(x):
    """
    Inverse of cotangent.
    """
    return atan(1 / x)

def acot2(y, x):
    """
    Inverse of cotangent, but takes into account the sign of each argument.
    """
    return atan2(x, y)

def csch(x):
    """
    Hyperbolic cosecant (reciprocal of hyperbolic sine).
    """
    return 1 / sinh(x)

def sech(x):
    """
    Hyperbolic secant (reciprocal of hyperbolic cosine).
    """
    return 1 / cosh(x)

def coth(x):
    """
    Hyperbolic cotangent (reciprocal of hyperbolic tangent).
    """
    return 1 / tanh(x)

def acsch(x):
    """
    Inverse of hyperbolic cosecant.
    """
    return asinh(1 / x)

def asech(x):
    """
    Inverse of hyperbolic secant.
    """
    return acosh(1 / x)

def acoth(x):
    """
    Inverse of hyperbolic cotangent.
    """
    return atanh(1 / x)

######################################################################
## misc.
######################################################################

def sgn(x):
    """
    Returns the sign of x:
    +1 if positive
    0 if zero
    -1 if negative
    """
    if x == 0:
        return 0
    else:
        return abs(x) / x

def cbrt(x):
    """
    Returns the cube root of x. Identical to x**(1.0/3), but with the
    useful addition that if x is negative, the function will still
    compute the correct (real) result, instead of raising a ValueError.
    """
    if x < 0:
        return (-1)*(-x)**(1.0/3)
    else:
        return x**(1.0/3)

def ln(x):
    """
    Synonym for natural log.
    """
    return log(x)

def log2(x):
    """
    Returns the base-2 logarithm of x.
    """
    return log(x) / 0.6931471805599453

# common synonym
lg = log2
