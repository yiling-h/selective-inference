import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return(pickle.load(input))

def multi_process(f, sequence, n_cores):
    from multiprocessing import Pool
    pool = Pool(processes=n_cores)
    
    result = pool.map(f, sequence)
    pool.close()
    pool.join()
    return result 
