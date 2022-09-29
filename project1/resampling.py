from linfitUtils import SplitData, Data
import numpy as np

class Resampler:
    def __init__(self, split_data: SplitData, resample_count: int):
        self.main_data = split_data
        self.resampled_data = split_data.copy()
        self.iterations = resample_count
        self.remaining_resamples = resample_count

    def bootstrap(self) -> bool:
        if self.remaining_resamples == 0:
            return False
        
        n = self.main_data.train_shape()[0]
        idxs = np.random.randint(0, n, n)

        X, y = self.main_data.train_data
        self.resampled_data.train_data = Data(X[idxs, :], y[idxs])

        self.remaining_resamples -= 1
        return True

    def cross_validation(self) -> bool:
        if self.remaining_resamples == 0:
            return False

        if self.remaining_resamples == self.iterations:
            # We must recombine the training and test data

            X_train, y_train = self.main_data.train_data
            X_test, y_test = self.main_data.test_data
            ntr, m = X_train.shape
            nte, _ = X_test.shape

            self.X = np.zeros((ntr+nte, m))
            self.X[:ntr, :] = X_train
            self.X[ntr:, :] = X_test

            self.y = np.zeros(ntr+nte)
            self.y[:ntr] = y_train
            self.y[ntr:] = y_test

            shuffle_idxs = np.arange(ntr+nte)
            np.random.shuffle(shuffle_idxs)
            self.X = self.X[shuffle_idxs, :]
            self.y = self.y[shuffle_idxs]

        bic = len(self.y) // self.iterations
        i = self.remaining_resamples-1

        if i == 0:
            X_test = self.X[0:bic, :]
            y_test = self.y[0:bic]
            X_train = self.X[bic:, :]
            y_train = self.y[bic:]
        elif i == self.iterations-1:
            X_test = self.X[bic*i:, :]
            y_test = self.y[bic*i:]
            X_train = self.X[:bic*i, :]
            y_train =  self.y[:bic*i]
        else:
            X_test = self.X[bic*i:bic*(i+1)]
            y_test = self.y[bic*i:bic*(i+1)]
            X_train = np.append(self.X[bic*(i-1):bic*i, :], self.X[bic*(i+1):, :], axis=0)
            y_train = np.append(self.y[bic*(i-1):bic*i], self.y[bic*(i+1):])

        self.resampled_data = SplitData(Data(X_test, y_test), Data(X_train, y_train))

        self.remaining_resamples -= 1
        return True

if __name__ == "__main__":
    from linfitUtils import LinFit

    def f(x):
        term1 = 0.5*np.exp(-(9*x-7)**2/4.0)
        term2 = -0.2*np.exp(-(9*x-4)**2)
        return term1 + term2

    x_ray = np.linspace(0, 1, 10000)
    y_ray = f(x_ray)
    
    split_data = SplitData.from_1d_polynomial(3, x_ray, y_ray, normalize=True)
    linfit = LinFit(split_data)

    print(f"Comparing regression methods with n={len(x_ray)} datapoints:\n")

    MSE, _ = linfit.test_fit()
    print(f"OLS:\n  MSE={MSE}\n")

    # test bootstrap

    avg_MSE = 0
    resampler = Resampler(split_data, len(x_ray)-1)
    while resampler.bootstrap():
        linfit.split_data = resampler.resampled_data
        MSE, R2 = linfit.test_fit()
        avg_MSE += MSE
    avg_MSE /= (len(x_ray)-1)
    print(f"bootstrap:\n  MSE={avg_MSE}\n")

    # test cross-validation

    print("cross-validation:")
    for k in [3, 5, 10, 50]:
        avg_MSE = 0
        resampler = Resampler(split_data, k)
        while resampler.cross_validation():
            linfit.split_data = resampler.resampled_data
            MSE, R2 = linfit.test_fit()
            avg_MSE += MSE
        avg_MSE /= k
        print(f"  k={f'{k}:':<3} MSE={avg_MSE}")
