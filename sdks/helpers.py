import time

def timer(func):
    def wrapper(*args, **keyArgs):
        start = time.time()
        ret = func(*args, **keyArgs)
        end = time.time()
        delta = end - start

        print('Time elapsed = ', delta // 60, ' m ', round(delta%60, 2), ' s')

        return ret

    return wrapper

if __name__ == "__main__":
    @timer
    def test_timer():
        print('test begin')
        time.sleep(2)
        print('test end')

    test_timer()