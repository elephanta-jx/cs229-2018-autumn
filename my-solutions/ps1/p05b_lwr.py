import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train, y_train)

    # Get MSE value on the validation set

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y = lwr.predict(x_valid)

    mse = (y - y_valid) @ (y - y_valid) / np.shape(y)
    print("mse: ", mse) # 0.33
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data

    plt.figure()
    plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2)
    plt.plot(x_valid[:, -1], y, 'ro', linewidth=2)



    # plt.xlim(x_train[:, -1].min()-margin1, x_train[:, -1].max()+margin1)
    # plt.ylim(y.min()-margin2, y.max()+margin2)

    plt.xlabel('x1')
    plt.ylabel('y')

    save_path = "./output/p05b"
    if save_path is not None:
        plt.savefig(save_path)
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.data_x = x
        self.data_y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = np.shape(self.data_x)
        input_m, _ = np.shape(x)
        res = np.zeros(input_m)

        for a in range(input_m):
            w = np.zeros((m, m))
            for i in range(m):
                norm = np.linalg.norm(x[a] - self.data_x[i])
                w[i, i] = np.exp(- np.power(norm, 2) / (2 * np.power(self.tau, 2)))

            theta = np.linalg.inv(self.data_x.T @ w @ self.data_x) @ self.data_x.T @ w @ self.data_y
            res[a] = theta @ x[a]

        return res
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(tau=0.5,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv')