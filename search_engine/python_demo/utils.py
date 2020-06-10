import time


def last_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        last = end - start
        print("Lasts %.2fs" % last)
        return res

    return wrapper

@last_time
def add(x, y):
    ans = x + y
    for i in range(10):
        ans += i
        time.sleep(0.01)

    return ans


if __name__ == '__main__':
    ans = add(1, 3)
    print(ans)