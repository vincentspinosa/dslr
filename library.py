def mean(args):
    """function returning the mean of a list of arguments."""
    return sum(args)/len(args)

def median(args):
    """function returning the median of a list of arguments."""
    s_args = sorted(args)
    l = len(s_args)
    m = l//2
    if l % 2 == 0:
        return (s_args[m] + s_args[m - 1])/2
    else:
        return s_args[m]


def quartile(args):
    """function returning a list with the first and third quartiles
    of a list of arguments."""
    l = len(args)
    m = l//4
    s_args = sorted(args)
    if l % 2 == 0:
        q1 = s_args[m - 1] + ((s_args[m] -s_args[m - 1]) * 0.25)
    else:
        q1 = s_args[m]
    if l % 2 == 0:
        q3 = s_args[-m - 1] + ((s_args[-m] - s_args[-m - 1]) * 0.75)
    else:
        q3 = s_args[-m - 1]
    return [float(q1), float(q3)]

def std(args):
    """function returning the standard deviation of a list of arguments."""
    m = mean(args)
    e = sum([(value - m)**2 for value in args])/len(args)
    return pow(e, 0.5)

def min(args):
    """function returning the smalest value of a list of arguments."""
    mini = args[0]
    for value in args:
        if value < mini:
            mini = value
    return mini

def max(args):
    """function returning the highest value of a list of arguments."""
    maxi = args[0]
    for value in args:
        if value > maxi:
            maxi = value
    return maxi

