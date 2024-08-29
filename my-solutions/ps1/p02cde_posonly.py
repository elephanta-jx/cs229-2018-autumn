import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    log = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True, label_col='t')
    log.fit(x_train, y_train)
    
    x_test, y_test = util.load_dataset(test_path, add_intercept=True, label_col='t')
    util.plot(x_test, y_test, log.theta, save_path=pred_path_c.replace(".txt", ""))
    y = log.predict(x_test)

    np.savetxt(pred_path_c, y)


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    log = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    log.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    _, t_test = util.load_dataset(test_path, add_intercept=True, label_col='t')
    util.plot(x_test, t_test, log.theta, save_path=pred_path_d.replace(".txt", ""))
    y = log.predict(x_test)
    np.savetxt(pred_path_d, y)

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    
    x_v, y_v = util.load_dataset(valid_path, add_intercept=True)
    v_plus = x_v[y_v == 1]
    num, _ = np.shape(v_plus)

    y_vc = log.predict(v_plus)
    alpha = np.sum(y_vc) / num 

   # log.theta = log.theta / alpha

    
    util.plot(x_test, t_test, log.theta, save_path=pred_path_e.replace(".txt", ""), correction=alpha)
    res = y / alpha
    np.savetxt(pred_path_e, res)

    # *** END CODER HERE

if __name__ == "__main__":
    main(train_path='../data/ds3_train.csv',
        valid_path='../data/ds3_valid.csv',
        test_path='../data/ds3_test.csv',
        pred_path='output/p02X_pred.txt')