import time


def last_time(comment):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            last = end - start

            if comment is not None:
                print(comment, end=' ')
            print("lasts %.2fs\n" % last)
            return res
        return inner_wrapper
    return wrapper


@last_time(comment="total")
def add(x, y):
    ans = x + y
    for i in range(10):
        ans += i
        time.sleep(0.01)

    return ans


def intersect_sorted_list(a, b):
    ans = []
    i = 0
    j = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            ans.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return ans


if __name__ == '__main__':
    a = [1, 2, 3, 6]
    b = [2, 4, 5, 6, 7]

    #ans = intersect_sorted_list(a, b)
    #print(ans)
    ans = add(3, 5)
    print(ans)
