import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()

    theta = clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path)
    util.plot(x_eval, y_eval, theta, save_path=pred_path.replace(".txt", ""))

    y = clf.predict(x_eval)

    np.savetxt(pred_path, y)

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, _ = np.shape(x)
        count = 0
        for i in y:
            if i == 1:
                count += 1
        phi = count / m

        sum0 = 0
        sum1 = 0
        for i in range(m):
            if y[i] == 0:
                sum0 += x[i]
            elif y[i] == 1:
                sum1 += x[i]

        mu0 = sum0 / (m - count)
        mu1 = sum1 / (m - count)


        sum = 0
        for i in range(m):
            if y[i] == 0:
                sum += np.outer(x[i] - mu0, x[i] - mu0)
            elif y[i] == 1:
                sum += np.outer(x[i] - mu1, x[i] - mu1)
        
        sigma = sum / m #this is wrong
        sig_inv = np.linalg.inv(sigma)

        self.theta = np.matmul((mu1 - mu0), sig_inv)
        self.theta0 = 0.5 * (mu0 @ sig_inv @ mu0 - mu1 @ sig_inv @ mu1) + np.log(phi / (1 - phi)) 

        res = np.insert(self.theta, 0, self.theta0)
        self.res = res
        util.plot(x, y, res, save_path="./")
        return res
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.array([1 / (1 + np.exp(-(self.theta @ xrow + self.theta0))) for xrow in x])
        
        # *** END CODE HERE


if __name__ == "__main__":
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01e_pred_2.txt')