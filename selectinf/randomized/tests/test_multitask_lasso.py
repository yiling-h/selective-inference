import numpy as np
from ..multitask_lasso import multi_task_lasso
from ...tests.instance import gaussian_multitask_instance

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

    print(multi_lasso.fit())


def test_multitask_lasso(ntask=5,
                         nsamples=500 * np.ones(5),
                         p=100,
                         global_sparsity=.9,
                         task_sparsity=.6,
                         sigma=1.*np.ones(5),
                         signal=np.array([0.3,5.]),
                         rhos=0.*np.ones(5),
                         weight=2.):

    nsamples = nsamples.astype(int)
    response_vars, predictor_vars, beta = gaussian_multitask_instance(ntask,
                                                                      nsamples,
                                                                      p,
                                                                      global_sparsity,
                                                                      task_sparsity,
                                                                      sigma,
                                                                      signal,
                                                                      rhos)[:3]

    feature_weight = weight * np.ones(p)
    randomizer_scales = np.ones(ntask)

    multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                            response_vars,
                                            feature_weight,
                                            randomizer_scales = randomizer_scales)
    multi_lasso.fit()

    #print('True Beta',beta)
    #print('Multi-task Solution', multi_lasso.fit())

if __name__ == "__main__":
    test_multitask_lasso()