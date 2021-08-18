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

    coverage_posi = list(traj.f_get_from_runs(name='posi.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_posi = list(traj.f_get_from_runs(name='posi.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_posi = list(traj.f_get_from_runs(name='posi.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_posi = list(traj.f_get_from_runs(name='posi.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_posi = list(traj.f_get_from_runs(name='posi.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_posi = list(traj.f_get_from_runs(name='posi.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_posi = list(traj.f_get_from_runs(name='posi.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())
    msetarget_posi = list(traj.f_get_from_runs(name='posi.mse.target', fast_access=True, auto_load=True, shortcuts=False).values())
    msetruth_posi = list(traj.f_get_from_runs(name='posi.mse.truth', fast_access=True, auto_load=True, shortcuts=False).values())
    runtime_posi = list(traj.f_get_from_runs(name='posi.runtime', fast_access=True, auto_load=True, shortcuts=False).values())

    signal_fac = traj.f_get('signal_fac').f_get_range()

    cseed = list(traj.f_get_from_runs(name='cseed', fast_access=True, auto_load=True, shortcuts=False).values())

    df = pd.DataFrame({'coverage_naive': coverage_naive,
                       'coverage_split50': coverage_split50,
                       'coverage_split67': coverage_split67,
                       'coverage_split33': coverage_split33,
                       'coverage_posi': coverage_posi,
                       'length_naive': length_naive,
                       'length_split50': length_split50,
                       'length_split67': length_split67,
                       'length_split33': length_split33,
                       'length_posi': length_posi,
                       'nnz_naive': nnz_naive,
                       'nnz_split50': nnz_split50,
                       'nnz_split67': nnz_split67,
                       'nnz_split33': nnz_split33,
                       'nnz_posi': nnz_posi,
                       'tp_naive': tp_naive,
                       'tp_split50': tp_split50,
                       'tp_split67': tp_split67,
                       'tp_split33': tp_split33,
                       'tp_posi': tp_posi,
                       'tn_naive': tn_naive,
                       'tn_split50': tn_split50,
                       'tn_split67': tn_split67,
                       'tn_split33': tn_split33,
                       'tn_posi': tn_posi,
                       'fp_naive': fp_naive,
                       'fp_split50': fp_split50,
                       'fp_split67': fp_split67,
                       'fp_split33': fp_split33,
                       'fp_posi': fp_posi,
                       'fn_naive': fn_naive,
                       'fn_split50': fn_split50,
                       'fn_split67': fn_split67,
                       'fn_split33': fn_split33,
                       'fn_posi': fn_posi,
                       'cseed': cseed,
                       'msetarget_naive': msetarget_naive,
                       'msetarget_split50': msetarget_split50,
                       'msetarget_split67': msetarget_split67,
                       'msetarget_split33': msetarget_split33,
                       'msetarget_posi': msetarget_posi,
                       'msetruth_naive': msetruth_naive,
                       'msetruth_split50': msetruth_split50,
                       'msetruth_split67': msetruth_split67,
                       'msetruth_split33': msetruth_split33,
                       'msetruth_posi': msetruth_posi,
                       'runtime_naive': runtime_naive,
                       'runtime_split50': runtime_split50,
                       'runtime_split67': runtime_split67,
                       'runtime_split33': runtime_split33,
                       'runtime_posi': runtime_posi
                       })

    if isinstance(traj.signal_fac, tuple):
        df.insert(0, 'Signal_Upper', [x[1] for x in signal_fac])
    else:
        df.insert(0, 'Signal_Fac', signal_fac)

    return df
