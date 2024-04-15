import multiprocessing as mp


def init():
    global pool
    pool = mp.Pool(processes=mp.cpu_count() - 1)
