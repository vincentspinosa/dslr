def mean_(args):
    """Function returning the mean of a list of arguments."""
    return sum(args) / len(args)


def median_(args):
    """Function returning the median of a list of arguments."""
    s_args = sorted(args)
    l = len(s_args)
    m = l // 2
    if l % 2 == 0:
        return (s_args[m] + s_args[m - 1]) / 2
    else:
        return s_args[m]


def quantile_linear_(sorted_vals, p):
    """Uses linear interpolation to find the exact quartile."""
    n = len(sorted_vals)
    if n == 0:
        return float('nan')
    # position in 1-based indexing: 1 + p*(n - 1)
    pos = 1 + p * (n - 1)
    # convert to 0-based index
    idx = pos - 1
    lower = int(idx)          # floor
    upper = min(lower + 1, n - 1)
    frac = idx - lower        # fractional part
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def quartile_(args):
    """Returns a list with quartiles 1 and 3."""
    s_args = sorted(args)
    q1 = quantile_linear_(s_args, 0.25)
    q3 = quantile_linear_(s_args, 0.75)
    return [float(q1), float(q3)]


def std_(args):
    """Function returning the standard deviation of a list of arguments."""
    m = mean_(args)
    variance = sum([(x - m) ** 2 for x in args]) / len(args)
    return variance ** 0.5


def min_(args):
    """Function returning the minimum value of a list of arguments."""
    x = float('inf')
    for value in args:
        if value < x:
            x = value
    return x


def max_(args):
    """Function returning the maximum value of a list of arguments."""
    x = float('-inf')
    for value in args:
        if value > x:
            x = value
    return x
