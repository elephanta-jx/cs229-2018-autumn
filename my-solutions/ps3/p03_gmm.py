import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    np.random.shuffle(x)
    groups = np.split(x, K)
    
    mu = np.mean(groups, axis=1)
    
    sigma = np.array([np.cov(each, rowvar=False) for each in groups])

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, 1.0 / K)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    m = x.shape[0]
    w = np.full((m, K), 1.0 / K)

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)
    
    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w

        const_term = np.array(1.0 / (np.power(2 * np.pi, 2 / 2) * np.power(np.linalg.det(sigma), 0.5)))
        for i, _ in enumerate(w):    
            exponent = np.exp(-0.5 * (x[i] - mu).reshape(4, 1, 2) @ np.linalg.inv(sigma) @ (x[i] - mu).reshape(4, 2, 1)).flatten()
            p_xz = const_term * exponent * phi
            w[i] = p_xz / np.sum(p_xz)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.mean(w, axis=0)

        a = np.sum(w.T.reshape(4, -1, 1) * x, axis=1)
        b = np.sum(w, axis=0).reshape(-1, 1)
        mu = a / b
             

        diff = (x - mu.reshape(4, 1, 2)).reshape(4, -1, 2, 1)
        a = diff @ np.transpose(diff, (0, 1, 3, 2)) 
        b = w.T

        
        sigma = np.sum(b.reshape(4, -1, 1, 1) * a, axis=1) / np.sum(b, axis=1).reshape(4, 1, 1)
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        
        # the log-likelihood function should not involve with w
        # and this is how i get it wrong 
        # same thing for the semi-supervised model
        a = np.array(1.0 / (np.power(2 * np.pi, 2 / 2) * np.power(np.linalg.det(sigma), 0.5)))
        diff = x - mu.reshape(4, 1, 2)
        b = np.exp(-0.5 * diff.reshape(4, -1, 1, 2) @ np.linalg.inv(sigma).reshape(4, 1, 2, 2) @ diff.reshape(4, -1, 2, 1)).reshape(4, -1).T
        
        
        c = np.log(a * b * phi / w) * w
        prev_ll = ll
        ll = np.sum(c)
        
        it += 1
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***
    
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w

        const_term = np.array(1.0 / (np.power(2 * np.pi, 2 / 2) * np.power(np.linalg.det(sigma), 0.5)))
        for i, _ in enumerate(w):    
            exponent = np.exp(-0.5 * (x[i] - mu).reshape(4, 1, 2) @ np.linalg.inv(sigma) @ (x[i] - mu).reshape(4, 2, 1)).flatten()
            p_xz = const_term * exponent * phi
            w[i] = p_xz / np.sum(p_xz)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        _, counts = np.unique(z, return_counts=True)
        a = np.sum(w, axis=0) + alpha * counts
        phi = a / (x.shape[0] + alpha * z.shape[0])

        a = np.sum(w.T.reshape(K, -1, 1) * x, axis=1)
        xt_groups = [x_tilde[z.astype(int).flatten() == j, :] for j in range(K)]
        b = alpha * np.sum(xt_groups, axis=1)
        c = np.sum(w, axis=0) + alpha * counts
        mu = (a + b) / c.reshape(-1, 1)

        x_diff = (x - mu.reshape(K, 1, 2)).reshape(K, -1, 2, 1)
        a = np.sum(w.T.reshape(K, -1, 1, 1) * x_diff @ np.transpose(x_diff, (0, 1, 3, 2)), axis=1)


        xt_diff = (xt_groups - mu.reshape(K, 1, 2)).reshape(K, -1, 2, 1)
        b = alpha * np.sum(xt_diff @ np.transpose(xt_diff, (0, 1, 3, 2)), axis=1)

        c = (np.sum(w, axis=0) + alpha * z.shape[0]).reshape(K, 1, 1) * np.eye(2)
        sigma = np.linalg.inv(c) @ (a + b) 

        
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        a = np.array(1.0 / (np.power(2 * np.pi, 2 / 2) * np.power(np.linalg.det(sigma), 0.5)))
        diff = (x - mu.reshape(4, 1, 2)).reshape(4, -1, 1, 2)
        b = np.exp(-0.5 * diff @ np.linalg.inv(sigma).reshape(4, 1, 2, 2) @ np.transpose(diff, (0, 1, 3, 2))).reshape(4, -1).T

        b_t = (-0.5 * diff @ np.linalg.inv(sigma).reshape(4, 1, 2, 2) @ np.transpose(diff, (0, 1, 3, 2))).reshape(4, -1).T
        
        # normalize w
        
        #w = np.where(w > 1e-3, w, 1e-3)
        with np.errstate(divide='ignore'):
            temp = np.log(w)
            temp[w == 0] = 0
        c_t = (np.log(a) + b_t + np.log(phi) - temp) * w
        
        l_unsup = np.sum(c_t)

        a = a.reshape(K, 1)
        b = np.exp(-0.5 * np.transpose(xt_diff, (0, 1, 3, 2)) @ np.linalg.inv(sigma).reshape(K, 1, 2, 2) @ xt_diff).reshape(4, -1) 
        c = np.log(a * b * phi.reshape(K, 1))
        l_sup = np.sum(c)

        prev_ll = ll
        ll = l_unsup + l_sup
        
        it += 1
        
        # the error is caused by that the summation should be inside log in semi-supervised learning
        # the log likelyhood is calculated in the wrong formula
        
        # *** END CODE HERE ***
    
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
