import pandas as pd


def make_df(traj):

    coverage_naive = list(traj.f_get_from_runs(name='naive.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_naive = list(traj.f_get_from_runs(name='naive.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_naive = list(traj.f_get_from_runs(name='naive.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_naive = list(traj.f_get_from_runs(name='naive.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_naive = list(traj.f_get_from_runs(name='naive.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_naive = list(traj.f_get_from_runs(name='naive.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_naive = list(traj.f_get_from_runs(name='naive.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_split50 = list(traj.f_get_from_runs(name='split0.5.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_split50 = list(traj.f_get_from_runs(name='split0.5.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_split50 = list(traj.f_get_from_runs(name='split0.5.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_split50 = list(traj.f_get_from_runs(name='split0.5.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_split67 = list(traj.f_get_from_runs(name='split0.67.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_split67 = list(traj.f_get_from_runs(name='split0.67.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_split67 = list(traj.f_get_from_runs(name='split0.67.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_split67 = list(traj.f_get_from_runs(name='split0.67.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    coverage_posi = list(traj.f_get_from_runs(name='posi.mean.coverage', fast_access=True, auto_load=True, shortcuts=False).values())
    length_posi = list(traj.f_get_from_runs(name='posi.mean.length', fast_access=True, auto_load=True, shortcuts=False).values())
    nnz_posi = list(traj.f_get_from_runs(name='posi.nonzero.nnz', fast_access=True, auto_load=True, shortcuts=False).values())
    tp_posi = list(traj.f_get_from_runs(name='posi.sigdet.tp', fast_access=True, auto_load=True, shortcuts=False).values())
    tn_posi = list(traj.f_get_from_runs(name='posi.sigdet.tn', fast_access=True, auto_load=True, shortcuts=False).values())
    fp_posi = list(traj.f_get_from_runs(name='posi.sigdet.fp', fast_access=True, auto_load=True, shortcuts=False).values())
    fn_posi = list(traj.f_get_from_runs(name='posi.sigdet.fn', fast_access=True, auto_load=True, shortcuts=False).values())

    signal_fac = traj.f_get('signal_fac').f_get_range()

    seed = list(traj.f_get_from_runs(name='seed', fast_access=True, auto_load=True, shortcuts=False).values())

    df = pd.DataFrame({'coverage_naive': coverage_naive,
                       'coverage_split50': coverage_split50,
                       'coverage_split67': coverage_split67,
                       'coverage_posi': coverage_posi,
                       'length_naive': length_naive,
                       'length_split50': length_split50,
                       'length_split67': length_split67,
                       'length_posi': length_posi,
                       'nnz_naive': nnz_naive,
                       'nnz_split50': nnz_split50,
                       'nnz_split67': nnz_split67,
                       'nnz_posi': nnz_posi,
                       'tp_naive': tp_naive,
                       'tp_split50': tp_split50,
                       'tp_split67': tp_split67,
                       'tp_posi': tp_posi,
                       'tn_naive': tn_naive,
                       'tn_split50': tn_split50,
                       'tn_split67': tn_split67,
                       'tn_posi': tn_posi,
                       'fp_naive': fp_naive,
                       'fp_split50': fp_split50,
                       'fp_split67': fp_split67,
                       'fp_posi': fp_posi,
                       'fn_naive': fn_naive,
                       'fn_split50': fn_split50,
                       'fn_split67': fn_split67,
                       'fn_posi': fn_posi,
                       'seed': seed,
                       })

    if isinstance(traj.signal_fac, tuple):
        df.insert(0, 'Signal_Upper', [x[1] for x in signal_fac])
    else:
        df.insert(0, 'Signal_Fac', signal_fac)

    return df
