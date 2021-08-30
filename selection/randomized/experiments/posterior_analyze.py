import pandas as pd


def make_df(traj):

    coverage_naive = list(traj.f_get_from_runs(name='naive.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_naive = list(traj.f_get_from_runs(name='naive.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_naive = list(traj.f_get_from_runs(name='naive.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_naive = list(traj.f_get_from_runs(name='naive.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_naive = list(traj.f_get_from_runs(name='naive.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_naive = list(traj.f_get_from_runs(name='naive.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_naive = list(traj.f_get_from_runs(name='naive.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_naive = list(traj.f_get_from_runs(name='naive.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_naive = list(traj.f_get_from_runs(name='naive.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_naive = list(traj.f_get_from_runs(name='naive.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_naive = list(traj.f_get_from_runs(name='naive.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_naive = list(traj.f_get_from_runs(name='naive.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_naive = list(traj.f_get_from_runs(name='naive.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_naive = list(traj.f_get_from_runs(name='naive.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_naive = list(traj.f_get_from_runs(name='naive.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_split50 = list(traj.f_get_from_runs(name='split0.5.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_split50 = list(traj.f_get_from_runs(name='split0.5.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_split50 = list(traj.f_get_from_runs(name='split0.5.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_split50 = list(traj.f_get_from_runs(name='split0.5.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_split50 = list(traj.f_get_from_runs(name='split0.5.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_split50 = list(traj.f_get_from_runs(name='split0.5.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_split50 = list(traj.f_get_from_runs(name='split0.5.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_split50 = list(traj.f_get_from_runs(name='split0.5.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_split50 = list(traj.f_get_from_runs(name='split0.5.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_split50 = list(traj.f_get_from_runs(name='split0.5.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_split50 = list(traj.f_get_from_runs(name='split0.5.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_split67 = list(traj.f_get_from_runs(name='split0.67.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_split67 = list(traj.f_get_from_runs(name='split0.67.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_split67 = list(traj.f_get_from_runs(name='split0.67.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_split67 = list(traj.f_get_from_runs(name='split0.67.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_split67 = list(traj.f_get_from_runs(name='split0.67.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_split67 = list(traj.f_get_from_runs(name='split0.67.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_split67 = list(traj.f_get_from_runs(name='split0.67.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_split67 = list(traj.f_get_from_runs(name='split0.67.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_split67 = list(traj.f_get_from_runs(name='split0.67.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_split67 = list(traj.f_get_from_runs(name='split0.67.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_split67 = list(traj.f_get_from_runs(name='split0.67.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_split33 = list(traj.f_get_from_runs(name='split0.33.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_split33 = list(traj.f_get_from_runs(name='split0.33.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_split33 = list(traj.f_get_from_runs(name='split0.33.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_split33 = list(traj.f_get_from_runs(name='split0.33.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_split33 = list(traj.f_get_from_runs(name='split0.33.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_split33 = list(traj.f_get_from_runs(name='split0.33.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_split33 = list(traj.f_get_from_runs(name='split0.33.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_split33 = list(traj.f_get_from_runs(name='split0.33.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_split33 = list(traj.f_get_from_runs(name='split0.33.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_split33 = list(traj.f_get_from_runs(name='split0.33.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_split33 = list(traj.f_get_from_runs(name='split0.33.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_split33 = list(traj.f_get_from_runs(name='split0.33.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_split33 = list(traj.f_get_from_runs(name='split0.33.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_split33 = list(traj.f_get_from_runs(name='split0.33.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_split33 = list(traj.f_get_from_runs(name='split0.33.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_posi50 = list(traj.f_get_from_runs(name='posi0.5.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_posi50 = list(traj.f_get_from_runs(name='posi0.5.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_posi50 = list(traj.f_get_from_runs(name='posi0.5.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_posi50 = list(traj.f_get_from_runs(name='posi0.5.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_posi50 = list(traj.f_get_from_runs(name='posi0.5.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_posi50 = list(traj.f_get_from_runs(name='posi0.5.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_posi50 = list(traj.f_get_from_runs(name='posi0.5.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_posi50 = list(traj.f_get_from_runs(name='posi0.5.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_posi50 = list(traj.f_get_from_runs(name='posi0.5.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_posi50 = list(traj.f_get_from_runs(name='posi0.5.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_posi50 = list(traj.f_get_from_runs(name='posi0.5.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_posi50 = list(traj.f_get_from_runs(name='posi0.5.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_posi50 = list(traj.f_get_from_runs(name='posi0.5.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_posi50 = list(traj.f_get_from_runs(name='posi0.5.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_posi50 = list(traj.f_get_from_runs(name='posi0.5.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_posi67 = list(traj.f_get_from_runs(name='posi0.67.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_posi67 = list(traj.f_get_from_runs(name='posi0.67.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_posi67 = list(traj.f_get_from_runs(name='posi0.67.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_posi67 = list(traj.f_get_from_runs(name='posi0.67.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_posi67 = list(traj.f_get_from_runs(name='posi0.67.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_posi67 = list(traj.f_get_from_runs(name='posi0.67.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_posi67 = list(traj.f_get_from_runs(name='posi0.67.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_posi67 = list(traj.f_get_from_runs(name='posi0.67.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_posi67 = list(traj.f_get_from_runs(name='posi0.67.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_posi67 = list(traj.f_get_from_runs(name='posi0.67.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_posi67 = list(traj.f_get_from_runs(name='posi0.67.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_posi67 = list(traj.f_get_from_runs(name='posi0.67.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_posi67 = list(traj.f_get_from_runs(name='posi0.67.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_posi67 = list(traj.f_get_from_runs(name='posi0.67.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_posi67 = list(traj.f_get_from_runs(name='posi0.67.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    signal_fac = traj.f_get('signal_fac').f_get_range()

    coverage_posi33 = list(traj.f_get_from_runs(name='posi0.33.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_posi33 = list(traj.f_get_from_runs(name='posi0.33.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_posi33 = list(traj.f_get_from_runs(name='posi0.33.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_posi33 = list(traj.f_get_from_runs(name='posi0.33.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_posi33 = list(traj.f_get_from_runs(name='posi0.33.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_posi33 = list(traj.f_get_from_runs(name='posi0.33.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_posi33 = list(traj.f_get_from_runs(name='posi0.33.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_posi33 = list(traj.f_get_from_runs(name='posi0.33.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_posi33 = list(traj.f_get_from_runs(name='posi0.33.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_posi33 = list(traj.f_get_from_runs(name='posi0.33.runtime', fast_access=True, auto_load=True, shortcuts=False).values())
    postcoverage_posi33 = list(traj.f_get_from_runs(name='posi0.33.postcoverage', fast_access=True, auto_load=True, shortcuts=False).values())
    posttp_posi33 = list(traj.f_get_from_runs(name='posi0.33.postsigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    posttn_posi33 = list(traj.f_get_from_runs(name='posi0.33.postsigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    postfp_posi33 = list(traj.f_get_from_runs(name='posi0.33.postsigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    postfn_posi33 = list(traj.f_get_from_runs(name='posi0.33.postsigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    cseed = list(traj.f_get_from_runs(name='cseed', fast_access=True, auto_load=True, shortcuts=False).values())

    df = pd.DataFrame({'coverage_naive': coverage_naive,
                       'coverage_split50': coverage_split50,
                       'coverage_split67': coverage_split67,
                       'coverage_split33': coverage_split33,
                       'coverage_posi50': coverage_posi50,
                       'coverage_posi67': coverage_posi67,
                       'coverage_posi33': coverage_posi33,
                       'length_naive': length_naive,
                       'length_split50': length_split50,
                       'length_split67': length_split67,
                       'length_split33': length_split33,
                       'length_posi50': length_posi50,
                       'length_posi67': length_posi67,
                       'length_posi33': length_posi33,
                       'nnz_naive': nnz_naive,
                       'nnz_split50': nnz_split50,
                       'nnz_split67': nnz_split67,
                       'nnz_split33': nnz_split33,
                       'nnz_posi50': nnz_posi50,
                       'nnz_posi67': nnz_posi67,
                       'nnz_posi33': nnz_posi33,
                       'tp_naive': tp_naive,
                       'tp_split50': tp_split50,
                       'tp_split67': tp_split67,
                       'tp_split33': tp_split33,
                       'tp_posi50': tp_posi50,
                       'tp_posi67': tp_posi67,
                       'tp_posi33': tp_posi33,
                       'tn_naive': tn_naive,
                       'tn_split50': tn_split50,
                       'tn_split67': tn_split67,
                       'tn_split33': tn_split33,
                       'tn_posi50': tn_posi50,
                       'tn_posi67': tn_posi67,
                       'tn_posi33': tn_posi33,
                       'fp_naive': fp_naive,
                       'fp_split50': fp_split50,
                       'fp_split67': fp_split67,
                       'fp_split33': fp_split33,
                       'fp_posi50': fp_posi50,
                       'fp_posi67': fp_posi67,
                       'fp_posi33': fp_posi33,
                       'fn_naive': fn_naive,
                       'fn_split50': fn_split50,
                       'fn_split67': fn_split67,
                       'fn_split33': fn_split33,
                       'fn_posi50': fn_posi50,
                       'fn_posi67': fn_posi67,
                       'fn_posi33': fn_posi33,
                       'posttp_naive': posttp_naive,
                       'posttp_split50': posttp_split50,
                       'posttp_split67': posttp_split67,
                       'posttp_split33': posttp_split33,
                       'posttp_posi50': posttp_posi50,
                       'posttp_posi67': posttp_posi67,
                       'posttp_posi33': posttp_posi33,
                       'posttn_naive': posttn_naive,
                       'posttn_split50': posttn_split50,
                       'posttn_split67': posttn_split67,
                       'posttn_split33': posttn_split33,
                       'posttn_posi50': posttn_posi50,
                       'posttn_posi67': posttn_posi67,
                       'posttn_posi33': posttn_posi33,
                       'postfp_naive': postfp_naive,
                       'postfp_split50': postfp_split50,
                       'postfp_split67': postfp_split67,
                       'postfp_split33': postfp_split33,
                       'postfp_posi50': postfp_posi50,
                       'postfp_posi67': postfp_posi67,
                       'postfp_posi33': postfp_posi33,
                       'postfn_naive': postfn_naive,
                       'postfn_split50': postfn_split50,
                       'postfn_split67': postfn_split67,
                       'postfn_split33': postfn_split33,
                       'postfn_posi50': postfn_posi50,
                       'postfn_posi67': postfn_posi67,
                       'postfn_posi33': postfn_posi33,
                       'msetarget_naive': msetarget_naive,
                       'msetarget_split50': msetarget_split50,
                       'msetarget_split67': msetarget_split67,
                       'msetarget_split33': msetarget_split33,
                       'msetarget_posi50': msetarget_posi50,
                       'msetarget_posi67': msetarget_posi67,
                       'msetarget_posi33': msetarget_posi33,
                       'msetruth_naive': msetruth_naive,
                       'msetruth_split50': msetruth_split50,
                       'msetruth_split67': msetruth_split67,
                       'msetruth_split33': msetruth_split33,
                       'msetruth_posi50': msetruth_posi50,
                       'msetruth_posi67': msetruth_posi67,
                       'msetruth_posi33': msetruth_posi33,
                       'runtime_naive': runtime_naive,
                       'runtime_split50': runtime_split50,
                       'runtime_split67': runtime_split67,
                       'runtime_split33': runtime_split33,
                       'runtime_posi50': runtime_posi50,
                       'runtime_posi67': runtime_posi67,
                       'runtime_posi33': runtime_posi33,
                       'postcoverage_naive': postcoverage_naive,
                       'postcoverage_split50': postcoverage_split50,
                       'postcoverage_split67': postcoverage_split67,
                       'postcoverage_split33': postcoverage_split33,
                       'postcoverage_posi50': postcoverage_posi50,
                       'postcoverage_posi67': postcoverage_posi67,
                       'postcoverage_posi33': postcoverage_posi33,
                       'cseed': cseed
                       })

    if isinstance(traj.signal_fac, tuple):
        df.insert(0, 'Signal_Upper', [x[1] for x in signal_fac])
    else:
        df.insert(0, 'Signal_Fac', signal_fac)

    return df
