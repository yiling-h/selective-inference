import numpy as np
from ..multitask_lasso import multi_task_lasso

def main():

    K = 4
    sample_sizes = (200, 200, 200, 200)
    p = 10
    beta = np.random.random((p, K))

    global_sparsity_rate = .90
    task_sparsity_rate = .50
    global_zeros = np.random.choice(p,int(round(global_sparsity_rate*p)))

    beta[global_zeros,:] = np.zeros((K,))
    for i in np.delete(range(p),global_zeros):
        beta[i,np.random.choice(K,int(round(task_sparsity_rate * K)))] = 0.
    print("beta ", beta)

    predictor_vars = {i: np.random.random((sample_sizes[i], p)) for i in range(K)}
    response_vars = {i: np.dot(predictor_vars[i], beta[:, i]) for i in range(K)}
    feature_weight = 1.25 * np.ones(p)
    randomizer_scales = np.ones(K)

    multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                            response_vars,
                                            feature_weight,
                                            randomizer_scales = randomizer_scales)

    print(multi_lasso.multitasking_solver())

if __name__ == "__main__":
    main()