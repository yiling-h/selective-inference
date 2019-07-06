import numpy as np, os

def generate_data(n, p, sigma=1., rho=0., V= 2.5, scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        scalingX = (X.std(0)[None, :] * np.sqrt(n))
        X /= scalingX

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.80:
            beta_true[i] = np.random.laplace(loc=0., scale= 0.10)
        else:
            beta_true[i] = np.random.laplace(loc=0., scale= V)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta* sigma, sigma, scalingX


def generate_data_instance(n, p, sigma=1., rho=0., s= 50, scale=True, center=True):
    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        scalingX = (X.std(0)[None, :] * np.sqrt(n))
        X /= scalingX

    beta_true = np.zeros(p)
    strong = []
    null = []
    u = np.random.uniform(0., 1., s)
    for i in range(s):
        if u[i] <= 0.80:
            null.append(np.random.laplace(loc=0., scale=0.5))
        else:
            strong.append(np.random.laplace(loc=0., scale=2.5))
    strong = np.asarray(strong)
    null = np.asarray(null)
    position = np.linspace(0, p-1, num=strong.shape[0], dtype=np.int)
    position_bool = np.zeros(p, np.bool)
    position_bool[position] = 1
    beta_true[position_bool] = strong

    vec_indx = np.arange(p)
    beta_true[np.random.choice(vec_indx[position_bool==0], null.shape[0])] = null
    Y = (X.dot(beta_true) + np.random.standard_normal(n)) * sigma

    return X, Y, beta_true * sigma, sigma, scalingX


def generate_signals_clusters(inpath, V, null_prob, detection_frac = 1., false_frac= 0.18, sigma=1.):

    X = np.load(os.path.join(inpath + "X.npy"))
    n, p = X.shape

    clusters = np.load(os.path.join(inpath + "clusters.npy")).astype(int)
    cluster_size = np.load(os.path.join(inpath + "cluster_size.npy")).astype(int)

    X = X[:,clusters]
    X -= X.mean(0)[None, :]
    scalingX = (X.std(0)[None, :] * np.sqrt(n))
    X /= scalingX

    beta_true = np.zeros(p)
    position_bool = np.zeros(p, np.bool)
    detection_threshold = detection_frac * np.sqrt(2. * np.log(p))

    strong = []
    null = []
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= null_prob:
            sig = np.random.laplace(loc=0., scale=0.10)
            if sig > detection_threshold:
                strong.append(sig)
            else:
                null.append(sig)
        else:
            sig = np.random.laplace(loc=0., scale=V)
            if sig > detection_threshold:
                strong.append(sig)
            else:
                null.append(sig)

    strong = np.asarray(strong)
    null = np.asarray(null)
    true_clusters = []

    cluster_length = np.cumsum(cluster_size)

    if strong.shape[0] <= cluster_size.shape[0]:
        cluster_choice = np.random.choice(cluster_size.shape[0], strong.shape[0], replace=False)

        for j in range(strong.shape[0]):
            pos_wcluster = np.random.choice(cluster_size[cluster_choice[j]], 1)
            if cluster_choice[j] > 0:
                beta_true[cluster_length[cluster_choice[j] - 1] + pos_wcluster] = strong[j]
                position_bool[cluster_length[cluster_choice[j] - 1] + pos_wcluster] = 1
                if strong[j] > detection_threshold:
                    true_clusters.append(
                        cluster_length[cluster_choice[j] - 1] + np.arange(cluster_size[cluster_choice[j]]))
            else:
                beta_true[pos_wcluster] = strong[j]
                position_bool[pos_wcluster] = 1
                if strong[j] > detection_threshold:
                    true_clusters.append(np.arange(cluster_size[cluster_choice[j]]))

    beta_true[~position_bool] = null
    Y = (X.dot(beta_true) + np.random.standard_normal(n)) * sigma

    cluster_list = []
    false_clusters = []
    for k in range(cluster_size.shape[0]):
        if k==0:
            clust_ind = clusters[:cluster_length[k]]
            cluster_list.append(clust_ind)
            if max(np.abs(beta_true[clust_ind])) < false_frac:
                false_clusters.append(clust_ind)
        else:
            clust_ind = clusters[cluster_length[k-1]:cluster_length[k]]
            cluster_list.append(clust_ind)
            if max(np.abs(beta_true[clust_ind])) < false_frac:
                false_clusters.append(clust_ind)

    return X, Y, beta_true * sigma, sigma, true_clusters, false_clusters, cluster_list, detection_threshold

def generate_signals(inpath, V, null_prob, detection_frac = 1., false_frac= 0.05, sigma=1.):

    X = np.load(os.path.join(inpath + "X.npy"))
    n, p = X.shape

    clusters = np.load(os.path.join(inpath + "clusters.npy")).astype(int)
    cluster_size = np.load(os.path.join(inpath + "cluster_size.npy")).astype(int)

    X = X[:,clusters]
    X -= X.mean(0)[None, :]
    scalingX = (X.std(0)[None, :] * np.sqrt(n))
    X /= scalingX

    beta_true = np.zeros(p)
    position_bool = np.zeros(p, np.bool)
    detection_threshold = detection_frac * np.sqrt(2. * np.log(p))

    strong = []
    null = []
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= null_prob:
            sig = np.random.laplace(loc=0., scale=0.10)
            if sig > detection_threshold:
                strong.append(sig)
            else:
                null.append(sig)
        else:
            sig = np.random.laplace(loc=0., scale=V)
            if sig > detection_threshold:
                strong.append(sig)
            else:
                null.append(sig)

    strong = np.asarray(strong)
    null = np.asarray(null)
    true_clusters = []

    cluster_length = np.cumsum(cluster_size)

    if strong.shape[0] <= cluster_size.shape[0]:
        cluster_choice = np.random.choice(cluster_size.shape[0], strong.shape[0], replace=False)

        for j in range(strong.shape[0]):
            pos_wcluster = np.random.choice(cluster_size[cluster_choice[j]], 1)
            if cluster_choice[j] > 0:
                beta_true[cluster_length[cluster_choice[j] - 1] + pos_wcluster] = strong[j]
                position_bool[cluster_length[cluster_choice[j] - 1] + pos_wcluster] = 1
                if strong[j] > detection_threshold:
                    true_clusters.append(
                        cluster_length[cluster_choice[j] - 1] + np.arange(cluster_size[cluster_choice[j]]))
            else:
                beta_true[pos_wcluster] = strong[j]
                position_bool[pos_wcluster] = 1
                if strong[j] > detection_threshold:
                    true_clusters.append(np.arange(cluster_size[cluster_choice[j]]))

    beta_true[~position_bool] = null
    true_signals = np.asarray([r for r in range(p) if position_bool[r]])
    false_signals = np.asarray([s for s in range(p) if (~position_bool[s] and np.fabs(beta_true[s])< false_frac)])
    Y = (X.dot(beta_true) + np.random.standard_normal(n)) * sigma

    return X, Y, beta_true * sigma, sigma, true_signals, false_signals, detection_threshold