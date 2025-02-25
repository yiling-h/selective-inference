from __future__ import division, print_function

import numpy as np
import nose.tools as nt
from nose.tools import nottest

import regreg.api as rr

from ..approx_reference_grouplasso import group_lasso
from ..paired_group_lasso_backup import paired_group_lasso
from ...tests.instance import gaussian_instance
from ...tests.flags import SET_SEED
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from ..randomization import randomization
from ...tests.decorators import rpy_test_safe

def test_augment():
    X = np.array([[1, 2, 8, 6],
                  [2, 4, 0, 1],
                  [9, 3, 2, 1]])
    pgl = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=1)

    X_aug = np.array([[2, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 8, 6, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 9, 2, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 2, 6, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 9, 3, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 8],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 2]])
    Y_aug = np.array([1,2,9,2,4,3,8,0,2,6,1,1])

    assert((pgl.X_aug == X_aug).all())
    assert((pgl.Y_aug == Y_aug).all())

def test_create_groups():
    X = np.array([[1, 2, 8, 6],
                  [2, 4, 0, 1],
                  [9, 3, 2, 1]])
    pgl = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=1)

    # b01 -> 0, b02 -> 1, b03 -> 2
    # b10 -> 0, b12 -> 3, b13 -> 4
    # b20 -> 1, b21 -> 3, b23 -> 5
    # b30 -> 2, b31 -> 4, b32 -> 5
    # groups = [b10, b20, b30, b01, b21, b31, b02, b12, b32, b03, b13, b23]
    #       = [0,1,2,0,3,4,1,3,5,2,4,5]
    assert(pgl.groups == [0,1,2,0,3,4,1,3,5,2,4,5])
    # 0: [0,1], 1: [0,2], 2: [0,3], 3: [1,2], 4: [1,3], 5: [2,3]
    assert(pgl.groups_to_vars[0] == [0,1] and pgl.groups_to_vars[1] == [0,2] and
           pgl.groups_to_vars[2] == [0,3] and pgl.groups_to_vars[3] == [1,2] and
           pgl.groups_to_vars[4] == [1,3] and pgl.groups_to_vars[5] == [2,3])
    for i in range(6):
        assert(pgl.weights[i] == 15)

    # Testing weights assignments
    weights = np.array([[0, 0, 1, 2],
                        [0, 0, 3, 4],
                        [1, 3, 0, 5],
                        [2, 4, 5, 0]])
    pgl2 = paired_group_lasso(X=X, weights=weights, ridge_term=0.0, randomizer_scale=1)
    for i in range(6):
        assert(pgl2.weights[i] == i)

def test_undo_vectorize():
    X = np.array([[1, 2, 8, 6],
                  [2, 4, 0, 1],
                  [9, 3, 2, 1]])
    pgl = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=1)
    # b01 -> 0, b02 -> 1, b03 -> 2
    # b10 -> 0, b12 -> 3, b13 -> 4
    # b20 -> 1, b21 -> 3, b23 -> 5
    # b30 -> 2, b31 -> 4, b32 -> 5
    # groups = [0,1,2,0,3,4,1,3,5,2,4,5]
    # vectorized parameters =
    #   [b10,b20,b30,b01,b21,b31,b02,b12,b32,b03,b13,b23]
    assert (pgl.undo_vectorize(0) == (1,0))
    assert (pgl.undo_vectorize(1) == (2,0))
    assert (pgl.undo_vectorize(2) == (3, 0))
    assert (pgl.undo_vectorize(3) == (0, 1))
    assert (pgl.undo_vectorize(4) == (2, 1))
    assert (pgl.undo_vectorize(5) == (3, 1))
    assert (pgl.undo_vectorize(6) == (0, 2))
    assert (pgl.undo_vectorize(7) == (1, 2))
    assert (pgl.undo_vectorize(8) == (3, 2))
    assert (pgl.undo_vectorize(9) == (0, 3))
    assert (pgl.undo_vectorize(10) == (1, 3))
    assert (pgl.undo_vectorize(11) == (2, 3))

def test_vec_to_mat():
    X = np.array([[1, 2, 8, 6],
                  [2, 4, 0, 1],
                  [9, 3, 2, 1]])
    pgl = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=1)
    # b01 -> 0, b02 -> 1, b03 -> 2
    # b10 -> 0, b12 -> 3, b13 -> 4
    # b20 -> 1, b21 -> 3, b23 -> 5
    # b30 -> 2, b31 -> 4, b32 -> 5
    # groups = [0,1,2,0,3,4,1,3,5,2,4,5]
    # vectorized parameters =
    #   [b10,b20,b30,b01,b21,b31,b02,b12,b32,b03,b13,b23]
    # To simplify testing, we use the index ij as the numerical values of entries
    mat = pgl.vec_to_mat(p=4, vec = [10,20,30,1,21,31,2,12,32,3,13,23])
    mat_true = np.array([[0,1,2,3],
                         [10,0,12,13],
                         [20,21,0,23],
                         [30,31,32,0]])
    assert((mat==mat_true).all())

def test_mat_to_vec():
    X = np.array([[1, 2, 8, 6],
                  [2, 4, 0, 1],
                  [9, 3, 2, 1]])
    pgl = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=1)
    # To simplify testing, we use the index ij as the numerical values of entries
    mat = np.array([[0,1,2,3],
                     [10,0,12,13],
                     [20,21,0,23],
                     [30,31,32,0]])
    vec = pgl.mat_to_vec(p=4, mat=mat)
    vec_true = [10,20,30,1,21,31,2,12,32,3,13,23]
    assert((vec == vec_true).all())

def test_paired_group_lasso(n=400,
                            p=100,
                            signal_fac=3,
                            s=5,
                            sigma=3,
                            target='full',
                            rho=0.4,
                            randomizer_scale=.75,
                            ndraw=100000):
    cov = np.array([[2, 1, 1, 1, 1],
                    [1, 2, 1, 1, 1],
                    [1, 1, 2, 1, 1],
                    [1, 1, 1, 2, 1],
                    [1, 1, 1, 1, 1]])
    X = np.random.multivariate_normal(mean=np.zeros((5,)), cov=cov, size=1000)
    """
    perturb = np.array([[0, -0.22352252, -0.40935027],
                    [-0.39547683, 0, 0.35188586],
                    [0.70123141, -0.4890036, 0]])
    
    perturb = np.zeros((3,3))
    """
    pgl = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=randomizer_scale)
    pgl.fit()

    X[:, [2, 0]] = X[:, [0, 2]]

    pgl2 = paired_group_lasso(X=X, weights=15.0, ridge_term=0.0, randomizer_scale=randomizer_scale)
    pgl2.fit()

def test_optimization_vars(n=400, p=15, randomizer_scale=.75):
    block = np.array([[2, 1, 1, 1, 1],
                    [1, 2, 1, 1, 1],
                    [1, 1, 2, 1, 1],
                    [1, 1, 1, 2, 1],
                    [1, 1, 1, 1, 1]])
    def gen_covariance(p):
        assert(p % 5 == 0)
        cov = np.zeros((p,p))
        for i in range(p // 5):
            cov[i*5:(i+1)*5, i*5:(i+1)*5] = block
        return cov

    # generate covariance and data
    cov = gen_covariance(p)
    X = np.random.multivariate_normal(mean=np.zeros((p,)), cov=cov, size=n)
    # Fit
    pgl = paired_group_lasso(X=X, weights=10.0, ridge_term=0.0, randomizer_scale=randomizer_scale)
    pgl.fit()
    T1 = - pgl.X_aug.T @ pgl.Y_aug
    T2 = pgl.opt_linear.dot(pgl.observed_opt_state)
    T3 = pgl.opt_offset
    assert(np.allclose(T1 + T2 + T3, pgl._initial_omega))
    assert(np.all(pgl.observed_opt_state > 0))

def test_selection_power(n=400, p=15, randomizer_scale=.75):
    block = np.array([[2, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1],
                      [1, 1, 2, 1, 1],
                      [1, 1, 1, 2, 1],
                      [1, 1, 1, 1, 1]])

    def gen_covariance(p):
        assert (p % 5 == 0)
        cov = np.zeros((p, p))
        for i in range(p // 5):
            cov[i * 5:(i + 1) * 5, i * 5:(i + 1) * 5] = block
        return cov

    # generate covariance and data
    cov = gen_covariance(p)
    prec = np.linalg.inv(cov)
    num_sparse_true = np.sum(prec == 0) - p

    avg_ratio = np.zeros((5,))
    weights = [1,5,10,15,20]

    for w in range(5):
        ratio = np.zeros((50,))
        for i in range(50):
            X = np.random.multivariate_normal(mean=np.zeros((p,)), cov=cov, size=n)
            pgl = paired_group_lasso(X=X, weights=weights[w], ridge_term=0.0, randomizer_scale=randomizer_scale)
            pgl.fit()
            num_sparse_est = np.sum(pgl.beta == 0) - p
            ratio[i] = num_sparse_est / num_sparse_true
        avg_ratio[w] = np.mean(ratio)

    import matplotlib.pyplot as plt
    plt.plot(weights, avg_ratio)
    plt.show()

def test_selection_consistency(n=400, p=15, randomizer_scale=.75):
    block = np.array([[2, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1],
                      [1, 1, 2, 1, 1],
                      [1, 1, 1, 2, 1],
                      [1, 1, 1, 1, 1]])

    def gen_covariance(p):
        assert (p % 5 == 0)
        cov = np.zeros((p, p))
        for i in range(p // 5):
            cov[i * 5:(i + 1) * 5, i * 5:(i + 1) * 5] = block
        return cov

    # generate covariance and data
    cov = gen_covariance(p)
    prec = np.linalg.inv(cov)
    print(prec)

    ## TODO: Do numerical experiments to see how robust the algorithm is against different randomizations
    ##       for the same X
    ## Step 1: Fix penalty
    ## Step 2: Generate one X object
    ## Step 3: Draw perturbations, and solve for the randomized problem
    ## Step 4: Calculate the proportion of zero entries that are always captured,
    ##         as well as the proportion of zero entries that are captured at least once
    ## Step 5: Repeat 2-4

def test_quadratic_form(n=400, p=15, randomizer_scale=.75):
    block = np.array([[2, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1],
                      [1, 1, 2, 1, 1],
                      [1, 1, 1, 2, 1],
                      [1, 1, 1, 1, 1]])

    def gen_covariance(p):
        assert (p % 5 == 0)
        cov = np.zeros((p, p))
        for i in range(p // 5):
            cov[i * 5:(i + 1) * 5, i * 5:(i + 1) * 5] = block
        return cov

    # generate covariance and data
    cov = gen_covariance(p=5)

    X = np.random.multivariate_normal(mean=np.zeros((5,)), cov=cov, size=5)
    # Fit
    pgl = paired_group_lasso(X=X, weights=1, ridge_term=0.0, randomizer_scale=1)
    pgl.fit()

def test_XXE(n=400, p=15, randomizer_scale=.75):
    cov = np.array([[2, 1, 1, 1],
                    [1, 2, 1, 1],
                    [1, 1, 2, 1],
                    [1, 1, 1, 1]])

    X = np.random.multivariate_normal(mean=np.zeros((4,)), cov=cov, size=10)
    # Fit
    pgl = paired_group_lasso(X=X, weights=5, ridge_term=0.0, randomizer_scale=1)
    pgl.fit()
