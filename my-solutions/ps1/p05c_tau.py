import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    def my_plot(x, pred_y, true_y, save_path):
        plt.figure()
        plt.plot(x[:, -1], true_y, 'bx', linewidth=2)
        plt.plot(x[:, -1], pred_y, 'ro', linewidth=2)

        plt.xlabel('x1')
        plt.ylabel('y')

        if save_path is not None:
            plt.savefig(save_path)

    # Search tau_values for the best tau (lowest MSE on the validation set)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    mse = float('inf')
    tau = 0
    

    for test in tau_values:
        lwr = LocallyWeightedLinearRegression(test)
        lwr.fit(x_train, y_train)

        y = lwr.predict(x_valid)
        path = f"./output/p05c_{test}.png"
        my_plot(x_valid, y, y_valid, path)
        mse_temp = (y - y_valid) @ (y - y_valid) / np.shape(y)
        if mse_temp < mse:
            mse = mse_temp
            tau = test

    # Fit a LWR model with the best tau value
    lwr_final = LocallyWeightedLinearRegression(tau)
    
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    lwr_final.fit(x_train, y_train)
    y = lwr_final.predict(x_test)

    my_plot(x_test, y, y_test, "./output/p05c_test")
    mse = (y - y_test) @ (y - y_test) / np.shape(y)
    print("test mse: ", mse) # 0.017
    # Save predictions to pred_path
    np.savetxt(pred_path, y)
    # Plot data

    

    

    # *** END CODE HERE ***


if __name__ == "__main__":
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='../data/ds5_train.csv',
         valid_path='../data/ds5_valid.csv',
         test_path='../data/ds5_test.csv',
         pred_path='output/p05c_pred.txt')