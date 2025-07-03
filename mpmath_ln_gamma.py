# Alternative approach: High-precision ln_gamma using Python's mpmath
# This can be used as a fallback when rug is not available

import mpmath
import json
import sys

def high_precision_ln_gamma(x, precision=50):
    """
    Calculate ln_gamma with arbitrary precision using mpmath
    
    Args:
        x: Input value (can be string for exact representation)
        precision: Number of decimal places of precision
    
    Returns:
        String representation of ln_gamma(x) with specified precision
    """
    # Set precision for mpmath
    mpmath.mp.dps = precision
    
    # Convert input to mpmath number
    if isinstance(x, str):
        x_mp = mpmath.mpf(x)
    else:
        x_mp = mpmath.mpf(x)
    
    # Calculate ln_gamma
    result = mpmath.loggamma(x_mp)
    
    # Return as string to preserve precision
    return str(result)

if __name__ == "__main__":
    # Read input from command line
    if len(sys.argv) != 3:
        print("Usage: python mpmath_ln_gamma.py <value> <precision>")
        sys.exit(1)
    
    value = sys.argv[1]
    precision = int(sys.argv[2])
    
    result = high_precision_ln_gamma(value, precision)
    print(result)
