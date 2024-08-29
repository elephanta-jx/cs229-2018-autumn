import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    log = LogisticRegression()
    log.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    util.plot(x_eval, y_eval, log.theta, save_path=pred_path.replace(".txt", ""))
    y = log.predict(x_eval)

    np.savetxt(pred_path, y)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # get Hessian Matrix 
        m, n = np.shape(x)

        theta = np.zeros(n)
        
        # update theta
        aaa  = 0 
        for _ in range(100):
            aaa += 1
            h = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    sum = 0
                    for k in range(m):
                        gz = 1 / (1 + np.exp(-(np.dot(theta, x[k]))))
                        sum += x[k][i] * x[k][j] * gz * (1- gz)
                    h[i][j] = sum / m    

            dtheta = np.zeros(n)
            for i in range(n):
                sum = 0
                for k in range(m):
                    gz = 1 / (1 + np.exp(-(np.dot(theta, x[k]))))
                    sum += x[k][i] * (y[k] - gz)
                dtheta[i] = sum / (-m)
            
            theta_k = theta -  np.matmul(np.linalg.inv(h), dtheta)
            if np.linalg.norm(theta - theta_k, ord=1) < 1e-5:
                theta = theta_k
                break
            theta = theta_k
        util.plot(x, y, theta, save_path="./")

        # print(theta)
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return np.array([1 / (1 + (np.exp(-(self.theta @ row)))) for row in x])

        # *** END CODE HERE ***

if __name__ == "__main__":
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01b_pred_2.txt')