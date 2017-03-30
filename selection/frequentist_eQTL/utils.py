import dill as pickle
import sys

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Saved pickle object to {}\n".format(filename))

def load_object(filename):
    with open(filename, 'rb') as input:
        return(pickle.load(input))
    sys.stderr.write("Loaded pickle object from {}\n".format(filename))

def multi_process(f, sequence, n_cores):
    sys.stderr.write("Using {} cores.\n".format(n_cores))
    from multiprocessing import Pool
    pool = Pool(processes=n_cores)
    result = pool.map(f, sequence)
    pool.close()
    pool.join()
    return result 
