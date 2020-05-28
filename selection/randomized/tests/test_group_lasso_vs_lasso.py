import numpy as np

from selection.randomized.group_lasso import group_lasso
from selection.tests.instance import gaussian_group_instance, gaussian_instance

def test_lasso_vs_glasso(n=500,
                     p=50,
                     signal_fac=3,
                     sgroup=50,
                     groups=np.arange(50),
                     sigma=3,
                     target='selected',
                     rho=0.1,
                     randomizer_scale=1):

    inst = gaussian_group_instance
    signal = np.sqrt(signal_fac * log(p))

    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      sgroup=sgroup,
                      groups=groups,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)

    weights = dict([(i, sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    
    #get a common randomization vector
    omega = randomization.isotropic_gaussian((p,), randomizer_scale * sigma_).sample()
    # run with lasso solver
    conv_lasso = group_lasso.gaussian(X,
                                Y,
                                groups,
                                weights,
                                randomizer_scale=randomizer_scale * sigma_,
                                use_lasso=True,
                                perturb = omega)

    _,soln_lasso = conv_lasso.fit()
    # run with group lasso solver
    conv_glasso = group_lasso.gaussian(X,
                                Y,
                                groups,
                                weights,
                                randomizer_scale=randomizer_scale * sigma_,
                                use_lasso=False,
                                perturb = omega)

    _,soln_glasso = conv_glasso.fit()
    #nonzero = conv.selection_variable['directions'].keys()
    soln_lasso_nz = np.column_stack((np.where(soln_lasso!=0)[0],soln_lasso[soln_lasso!=0]))
    soln_glasso_nz = np.column_stack((np.where(soln_glasso!=0)[0],soln_lasso[soln_glasso!=0]))

    print("Check solvers", np.allclose(soln_lasso,soln_glasso,rtol=1e-03))
    print("lasso:",soln_lasso_nz)
    print("group lasso:",soln_glasso_nz)
    
test_lasso_vs_glasso()
