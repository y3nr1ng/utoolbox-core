
def is_optimal_size(n, factors=(2, 3, 5, 7)):
    n = int(n)
    assert n > 0, "size must be a positive integer"
    for factor in factors:
        while n % factor == 0:
            n /= factor
    return n == 1

def find_optimal_size(target, prefer_pos=True):
    if is_optimal_size(target):
        return target
    else:
        for abs_delta in range(1, target):
            sign = 1 if prefer_pos else -1
            for delta in (sign*abs_delta, -sign*abs_delta):
                candidate = target + delta
                if is_optimal_size(candidate):
                    return candidate

if __name__ == '__main__':
    target = 1610
    optimal = find_optimal_size(target, prefer_pos=False)
    print("target={}, optimal={}".format(target, optimal))
