
# %% Load Libraries

import gpflow;
import tensorflow as tf;
import tensorflow_probability as tfp;
import numpy as np;
import matplotlib.pyplot as plt;
from scipy import stats;
import pandas as pd;
import datetime
# import sklearn 
from sklearn.utils.multiclass import unique_labels;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import OneHotEncoder;
from sklearn import preprocessing;
from sklearn.metrics import accuracy_score, mean_squared_error;
from sklearn.linear_model import Ridge, LinearRegression;
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer;
from sklearn.pipeline import make_pipeline;
from tensorflow.python.framework.errors_impl import InvalidArgumentError;
import robustgp;
from robustgp import ConditionalVariance;
from bayesian_benchmarks.data import *;
from bayesian_benchmarks.data import _ALL_REGRESSION_DATATSETS;



# import scipy
import time;
# from ast import literal_eval
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits;
from tqdm import tqdm;
from scipy import stats;

#%% Utility Functions

class Stopwatch():
    """Class for a stopwatch timer.
    """

    def __init__(self):
        self.current_time = 0

    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.current_time += time.time()-self.start_time

    def reset(self):
        self.current_time = 0 
        pass    

    def read_time(self):
        return self.current_time

def meanpred_baseline(_, Y_train, __, Y_test):
    pf = np.mean(Y_train)
    pv = np.var(Y_train)
    elbo = np.sum(stats.norm.logpdf(Y_train, pf, pv ** 0.5))
    rmse = np.mean((Y_test - pf) ** 2.0) ** 0.5
    nlpp = -np.mean(stats.norm.logpdf(Y_test, pf, pv ** 0.5))
    return elbo, rmse, nlpp

def linear_baseline(X_train, Y_train, X_test, Y_test):
    reg = LinearRegression().fit(X_train, Y_train)
    residuals = reg.predict(X_train) - Y_train
    pred_var = np.var(residuals)

    elbo = np.sum(stats.norm.logpdf(residuals, scale=pred_var ** 0.5))

    residuals_test = reg.predict(X_test) - Y_test
    rmse = np.mean(residuals_test ** 2.0) ** 0.5
    nlpp = -np.mean(stats.norm.logpdf(residuals_test, scale=pred_var ** 0.5))

    return elbo, rmse, nlpp

def load_dataset(dataset="1D", y_hot_encoding=True, normalised=True, subsample=False, subsample_size=200):

    """
    Dataset loading utility function.

    Args:
        dataset (str, optional): the string of the dataset to be loaded. Defaults to "1D".

    Returns:
        Training and test outputs. One hot encoding for classification datasets.
    """

    # print(type(dataset))
    print("loading dataset:{}".format(dataset))


    def load_1D_dataset(N_total=3000, type="periodic"):
        """
        Function to load a customisable toy 1D regression dataset

        Args:
            N_total (int, optional): Total number of data points. Defaults to 200.
            test_set_proportion (float, optional): Proportion of test point to all data points. Defaults to 0.2.
        """
        rng = np.random.RandomState(123)
    
        
        
        
        
        def periodic_func(x):
            return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

        def exp_plus_cubic(x):
            return 50*np.power(x,3) - 8*np.square(x) - 1.5*x - 0.2*np.exp(x)

        N_tot = N_total  # Number of total input points (training and test)

        X = rng.rand(N_tot, 1) * 2 - 1  # X values
        if type=="periodic":
            Y = periodic_func(X) + 0.2 * rng.randn(N_tot, 1)  # Noisy Y values
        elif type=="exp+cubic":
            Y = exp_plus_cubic(X) + 5 * rng.randn(N_tot, 1)
        elif type=="exp+cubic+step":
            Y = exp_plus_cubic(X) + 5 * rng.randn(N_tot, 1) + 15*np.heaviside(X,0)
        else:
            raise ValueError("No 1D dataset of that kind.")
        
        
        Y = (Y-np.mean(Y))/np.std(Y) # Normalise dataset

        return  X, Y

    def load_uci_data(dataset):
        dataset = dataset.lower()
        if dataset == "wilson_elevators":
            data = Wilson_elevators()
        elif dataset == "wilson_energy":
            data = Wilson_energy()
        elif dataset == "wilson_gas":
            data = Wilson_gas()    
        elif dataset == "wilson_keggdirected":
            data = Wilson_keggdirected()
        elif dataset == "wilson_keggundirected":
            data = Wilson_keggundirected()
        elif dataset == "wilson_kin40k":
            data = Wilson_kin40k()
        elif dataset == "wilson_airfoil":
            data = Wilson_airfoil()
        elif dataset == "wilson_solar":
            data = Wilson_solar()
        elif dataset == "winered":
            data = WineRed()
        elif dataset == "winewhite":
            data = WineWhite()
        elif dataset == "wilson_yacht":
            data = Wilson_yacht()
        elif dataset == "wilson_concrete":
            data = Wilson_concrete()
        elif dataset == "wilson_wine":
            data = Wilson_wine()
        elif dataset == "wilson_skillcraft":
            data = Wilson_skillcraft()
        elif dataset == "wilson_parkinsons":
            data = Wilson_parkinsons()
        elif dataset == "wilson_pumadyn32nm":
            data = Wilson_pumadyn32nm()
        elif dataset == "wilson_sml":
            data = Wilson_sml()
        elif dataset == "power":
            data = Power()
        elif dataset == "naval":
            data = Naval()
        elif dataset == "kin8nm":
            data = Kin8mn()
        else:
            raise ValueError("Dataset not installed in 'load_uci_dataset' function.")

        return data

    if dataset=="mnist" or dataset=="fashion mnist":
        if dataset=="mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data();
        if dataset=="fashion mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data();
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)

        # preprocessing
        x_train = x_train.reshape(60000, 784)/255
        x_test = x_test.reshape(10000, 784)/255

    elif dataset[0:3]=="1D_":
        X, y = load_1D_dataset(type=dataset[3:])
        # Split data into training and test sets
        
        if normalised:
            y = (y-np.mean(y))/np.std(y) # Normalise the outputs
            X = preprocessing.normalize(X, norm='l2', axis=0)


        return train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=88,
            shuffle=True
            )

        
        
        
        

        
    elif dataset in _ALL_REGRESSION_DATATSETS:
        #from scipy.io import loadmat
        noise = False
        
        
        
        # if dataset[0:5] == "noisy":
        #     dataset =  dataset[6:]
        #     noise = True
        data = load_uci_data(dataset)
        X, y = data.read_data()
        if subsample:
            perm = np.random.RandomState(seed=4).permutation(len(y))
            X = X[perm[:subsample_size]]
            y = y[perm[:subsample_size]]
        if normalised:
            y = (y-np.mean(y))/np.std(y) # Normalise the outputs
            X = preprocessing.normalize(X, norm='l2', axis=0)
        if noise:
            y = y + np.random.normal(0, 0.0068, np.shape(y)) 
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
        return x_train, x_test, y_train, y_test

    

        
    # elif dataset in {"diabetes", "breast", "iris", "digits"}:
    #     if dataset=="diabetes":
    #         data = load_diabetes()
    #     if dataset=="breast":
    #         data = load_breast_cancer()
    #     if dataset=="iris":
    #         data = load_iris()
    #     if dataset=="digits":
    #         data = load_digits()


    #     y = data["target"]
    #     #y = np.array([float(y_one) for y_one in y])
    #     X = data["data"]
    #     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)


        
        # print(C)
    
    else:
        raise ValueError("Dataset not found.")


    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    N = len(y_train)

    if (dataset=="breast" or "iris" or "digits" or "mnist") and y_hot_encoding==True:

        C = len(unique_labels(y_train))
        y_hot = np.zeros((N, C))
        
        y_hot = OneHotEncoder(handle_unknown='ignore').fit_transform(y_train).toarray()

        # print("poo")

        #y_hot = y_hot.reshape(-1, 1)

        return x_train, x_test, y_hot, y_test

    else:

        D = len(x_train[0])

        return x_train, x_test, y_train, y_test

def lin_mod_scores(dataset:str, linear_model:str):
    x_train, x_test, y_train, y_test = load_dataset(dataset)

    if linear_model == "spline3":
        model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    elif linear_model=="linear":
        return linear_baseline(x_train, y_train, x_test, y_test)
    elif linear_model=="mean":
        return meanpred_baseline(x_train, y_train, x_test, y_test)
    else:
        raise ValueError("Model not found.")
    
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)

def make_kernel_obj(kernel_str, num_dim):#* Kernel
    """Makes a kernel object to be used in GP models (in gpflow)

    Args:
        kernel_str (str): Kernel name (see available ones in the class)
        num_dim (int): number of dimensions of input data

    Raises:
        ValueError: If kernel string is not valid

    Returns:
        gpflow.kernel.Kernel:  kernel object
    """
    class ShiftedArcCosine(gpflow.kernels.ArcCosine):
        def __init__(self, shift=0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shift = gpflow.Parameter(shift)

        def K(self, X, X2=None):
            return super().K(X - self.shift, X2 - self.shift if X2 is not None else X2)

        def K_diag(self, X):
            return super().K_diag(X - self.shift)

    if kernel_str == "SE":
        return gpflow.kernels.SquaredExponential()

    elif kernel_str == "ARD SE":
        return gpflow.kernels.SquaredExponential(lengthscales=[1.]*num_dim)
    
    elif kernel_str == "ARD Matern12":
        return gpflow.kernels.Matern12(lengthscales=[1.]*num_dim)
    elif kernel_str == "ARD Matern32":
        return gpflow.kernels.Matern32(lengthscales=[1.]*num_dim)

    elif kernel_str == "ARD ArcCosine":
        return ShiftedArcCosine(
            order=0,
            weight_variances=[1.]*num_dim
            )

    elif kernel_str == "ARD ArcCosine (order 1)":
        return ShiftedArcCosine(
            order=1,
            weight_variances=[1.]*num_dim
            )

    elif kernel_str == "ARD ArcCosine (order 2)":
        return ShiftedArcCosine(
            order=2,
            weight_variances=[1.]*num_dim
            )

    elif kernel_str == "ARD SE + ArcCosine":
        k1 = gpflow.kernels.ArcCosine(
            order=0,
            weight_variances=[1.]*num_dim
            )
        k2 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*num_dim)
        return k1 + k2

    elif kernel_str == "ARD SE + ArcCosine + Linear":
        k1 = gpflow.kernels.ArcCosine(
            order=0,
            weight_variances=[1.]*num_dim
            )
        k2 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*num_dim)
        k3 = gpflow.kernels.Linear(variance=[1.]*num_dim)
        return k1 + k2 + k3

    # elif kernel_str == "ARD ArcCosine + Exponential":
    #     k1 = gpflow.kernels.ArcCosine(
    #         order=0,
    #         active_dims=list(range(num_dim))
    #         )
    #     k2 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*num_dim)
    #     return k1 + k2

    elif kernel_str == "ARD ArcCosine + Polynomial (3)":
        k1 = gpflow.kernels.ArcCosine(
            order=0,
            weight_variances=[1.]*num_dim
            )
        k2 = gpflow.kernels.Polynomial(3, variance=[1.]*num_dim)
        return k1 + k2

    elif kernel_str == "Polynomial (3)":
        
        k2 = gpflow.kernels.Polynomial(3, variance=[1.]*num_dim)
        return k2



    else:
        raise ValueError("Kernel not supported.")

def resort_data(X_test, *args):
    """
    Function to resort data, such that they are in ascending order wrt X_test. Useful for plotting curves.

    Args:
        X_test (array)

    Returns:
        X_test reordered
    """


    X_test, *args = zip(*sorted(zip(X_test, *args)))

    X_test = np.array(X_test)
    
    args = [np.array(arg) for arg in args]

    return X_test, *args


#%%

def sgpr_regression_trainer(
    dataset,

    num_inducing_points,
    num_iterations,
    optimiser="L-BFGS-B",
    opt_learning_rate=None,

    *,
    kernel = "ARD SE",
    initial_inducing_point_locations="Greedy Variance",
    hyperparams_dict=dict(),
    initial_q_u=None,
    notes=None,
    
    opt_strategy = "retrain",
    output_plot_csv=False,
    reinit_freq = 25,
    parameter_logging=None,
    ):


    
    def make_assertions():
        if not type(num_inducing_points) is int:
            raise ValueError("Number of inducing points must be an integer.")
        if not type(num_iterations) is int:
            raise ValueError("Iterations must be an integer.")
        if minibatch_size is not None and not type(minibatch_size) is int:
            raise ValueError("Minibatch size must be an integer.")
        if type(initial_inducing_point_locations) != str and not len(initial_inducing_point_locations)==num_inducing_points:
            raise ValueError("Number of inducing points must match length of inducing locations array")
        if optimiser=="BFGS":
            assert minibatch_size is None and opt_learning_rate is None
        if minibatch_size is not None and model=="SGPR":
            raise ValueError("No minibatching allowed with SGPR.")
        # if retrain_freeZ and greedy_var_reinitialisation:
        #     raise InvalidArgumentError("You should not be implementing reinitialisation method and double retraining at the same time.")
    make_assertions()

    
    if notes is None:
        if opt_strategy=="reinit":
            assert reinit_freq is not None
            state = "reinitialised (every {} iterations)".format(reinit_freq)
        elif opt_strategy=="retrain":
            state = "fixed-->trainable"
        elif opt_strategy=="init":
            state = "fixed"
        elif opt_strategy=="train":
            state = "trainable"
        elif opt_strategy=='reinittrain':
            assert reinit_freq is not None
            state = "reinitialised ({}) then trainable".format(reinit_freq)
        else:
            raise InvalidArgumentError("Wrong input for optimisation strategy.")
        
        notes = "{} Dataset (M={}), using {} optimiser and {} kernel, and {} Z with {} initialisation.".format(dataset, num_inducing_points, optimiser, kernel, state, initial_inducing_point_locations)



    
    #- Start stopwatch to time setting up the model
    
    #* Stopwatch instance
    watch = Stopwatch()
    watch.reset()
    #* Start the Stopwatch
    watch.start()


    #- Set up data

    #* Load dataset into 4 arrays
    x_train, x_test, y_train, y_test = load_dataset(
        dataset=dataset,
        y_hot_encoding = False,
        normalised=True,
        )

    #* Convert training and test datasets into Tensorflow dataset objects
    original_training_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
        );
    original_test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
        );




    
    #- Configure settings for model

    num_training_datapoints = len(y_train)
    num_dim = len(x_train[0])

    #* Number of logs to make during optimisation
    n_logs = 1000

    #* Pick value for jitter
    jitter = 1e-5 # 1e-4
    gpflow.config.set_default_jitter(jitter); #adaptive jitter?
    # gpflow.config.set_default_positive_minimum(1.0e-5)


    
    #* Likelihood function for model
    likelihood = gpflow.likelihoods.Gaussian()
        
    kernel_obj = make_kernel_obj(kernel, num_dim)

    
    #* Initialisation of inducing points
    if initial_inducing_point_locations=="Random":
        Z = x_train[:num_inducing_points, :].copy()
    elif initial_inducing_point_locations=="Greedy Variance":
        Z_initer = ConditionalVariance()
        Z = Z_initer.compute_initialisation(x_train, num_inducing_points, kernel_obj)[0]

    else:
        Z = initial_inducing_point_locations



    #* q(u)
    if initial_q_u is not None:
        assert len(initial_q_u)==2
        q_mu=initial_q_u[0]
        q_sqrt=initial_q_u[1]
    else:
        q_mu=None
        q_sqrt=None





    #- Set up the model
   
    #* Initialise the model
 
    m = gpflow.models.SGPR(
        data=(x_train, y_train),
        inducing_variable=Z,
        kernel=kernel_obj,
        )
    

    
    #* Initialise hyperparams and set trainable ones.
    def initialise_hyperparameters(hyperparameter_dict):
        #* Initialise hyperparams:
        # # This is done using heuristics from https://infallible-thompson-49de36.netlify.app/

        if kernel == "ARD ArcCosine":
            pass


        if kernel in ["ARD SE", "ARD Matern12", "ARD Matern32"]:
            initial_kernel_lengthscale = np.std(x_train, axis=0)
            mean = np.mean(initial_kernel_lengthscale[initial_kernel_lengthscale > 0])
            initial_kernel_lengthscale[initial_kernel_lengthscale == 0] = mean
            m.kernel.lengthscales.assign(hyperparameter_dict.setdefault("kernel lengthscales", initial_kernel_lengthscale))
            
            m.kernel.variance.assign(hyperparameter_dict.setdefault("kernel variance", np.std(y_train)))

            m.likelihood.variance.assign(hyperparameter_dict.setdefault("likelihood variance", hyperparameter_dict["kernel variance"]/50))

            #* Constrain hyperparameters such that they do not go to wacky values during optimisation:
            if not kernel == "ARD ArcCosine":
                lengthscale_constrained_transform = tfp.bijectors.Sigmoid(
                    gpflow.utilities.to_default_float(min(m.kernel.lengthscales.numpy())/100),
                    gpflow.utilities.to_default_float(max(m.kernel.lengthscales.numpy())*100),
                )
            else:
                lengthscale_constrained_transform = None

            var_constrained_transform = tfp.bijectors.Sigmoid(
                gpflow.utilities.to_default_float(m.kernel.variance.numpy()/1000),
                gpflow.utilities.to_default_float(m.kernel.variance.numpy()*1000),
            )

            # noise_constrained_transform = tfp.bijectors.Sigmoid(
            #     gpflow.utilities.to_default_float(jitter*10), # minimum the noise can be is 10 times jitter.
            #     gpflow.utilities.to_default_float(m.likelihood.variance.numpy()), # max is the kernel variance
            # )
            
            # print(hyperparameter_dict)

            new_len = gpflow.Parameter(
                m.kernel.lengthscales.numpy(),transform=lengthscale_constrained_transform)
            
            new_var = gpflow.Parameter(
                m.kernel.variance.numpy(),
                transform = var_constrained_transform)
            
            # new_noi = gpflow.Parameter(
            #     m.likelihood.variance.numpy(),
            #     transform = noise_constrained_transform)
            
            m.kernel.lengthscales = new_len
            m.kernel.variance = new_var
            # m.likelihood.variance = new_noi

            return {param[1:].replace(".", " "): value.numpy() for (param, value) in gpflow.utilities.parameter_dict(m).items()}
        #// def set_fixed_parameters(parameters_set):
        #     if "inducing variable" in parameters_set:
        #         print("Fixing inducing variable. Not including as a trainable parameter.")
        #         gpflow.set_trainable(m.inducing_variable, False)
        #     if "likelihood variance" in parameters_set:
        #         gpflow.set_trainable(m.likelihood.variance, False)
        #     if "kernel variance" in parameters_set:
        #         gpflow.set_trainable(m.kernel.variance, False)
        #     if "kernel lengthscales" in parameters_set:
        #         gpflow.set_trainable(m.kernel.lengthscales, False)
        #     if "q_u" in parameters_set:
        #         gpflow.set_trainable(m.q_mu, False)
        # //        gpflow.set_trainable(m.q_sqrt, False)
    
    hyperparams_dict = initialise_hyperparameters(hyperparams_dict)

    
    
    #* Define training loss
    training_loss=m.training_loss_closure()


   
    
    #- Set up necessary instances and functions for training
        

    #* Counter for BFGS optimisation logging
    global iter_n
    iter_n = 0
    log_frequency = 25

    #* List for ELBO values during 
    elbo_logs = []
    rmse_logs = []
    nlpd_logs = []
    times = [] 

    def log_rmse(Y_predict_mu, Y_test):
        """Root mean squared error loss

        Args:
            Y_predict_mu
            Y_test

        Returns:
            rmse
        """
        rmse = np.sqrt(np.mean((Y_predict_mu - Y_test)**2))
        rmse_logs.append(rmse)
        return rmse

    def log_nlpd(Y_predict_mu, Y_predict_var, Y_test, X_test):
        """Loss function - negative log validation density loss.

        Args:
            Y_predict_mu (_type_): 
            Y_predict_var (_type_): 
            Y_test (_type_): 

        """

        term_1 = np.log(Y_predict_var)
        term_2 = ((Y_predict_mu - Y_test)**2)/(Y_predict_var)
        term_3 = np.log(2*np.pi)
        nlpd = np.sum(-0.5*(term_1+term_2+term_3))
        nlpd_per_dp = nlpd/len(Y_test)
        nlpp = -np.mean(m.predict_log_density((X_test, Y_test)))

        nlpd_logs.append(nlpp)

        return nlpp

    def log_opt(iteration):
        """
        Utility function to log the elbo score for each iteration during optimisaiton.

        Args:
            iteration number (just for printing)
        """
        #* Stop the watch before logging anything
        watch.stop()
        global iter_n

        if iter_n % log_frequency == 0:
            #* Calculate elbo
            if model == "SGPR":
                elbo =  m.elbo()
            elif model=="SVGP":
                elbo =  m.elbo((x_train, y_train))
            

            #* Make predictions
            y_pred_mu, y_pred_var = m.predict_y(x_test)
            
            #* Log results
            nlpd_loss = log_nlpd(y_pred_mu, y_pred_var, y_test, x_test)
            rmse_loss = log_rmse(y_pred_mu, y_test)
            elbo_logs.append(float(elbo))
            times.append(watch.read_time())
            if parameter_logging is not None:
                log_parameters(parameter_logging)

            #* Print optimisation stats to console
            if optimiser=="Adam" or optimiser=="SGD":
                tqdm.write("Iteration: {}, RMSE: {:.4e}, NLPD: {:.4f}, ELBO: {:.4e}".format(iteration, rmse_loss, nlpd_loss, elbo))
            elif optimiser in {"BFGS", "L-BFGS-B"}:
                print("Iteration: {}, RMSE: {:.4e}, NLPD: {:.4f}, ELBO: {:.4e}, Time: {}".format(iter_n, rmse_loss, nlpd_loss, elbo, watch.read_time()))

            if output_plot_csv:
                log_into_csv(watch.read_time(), float(elbo), rmse_loss, nlpd_loss)
        else: print("Iteration: {}".format(iter_n))

        #* Increment iteration counter
        iter_n += 1 

        #* Start stopwatch again
        watch.start()


    def run_optimizer(learning_rate, iterations):
        """Carries out optimisation of model's objective function

        Args:
            learning_rate (float)
            iterations (integer)
        """
        global iter_n
        #* Optimiser object
        if optimiser in ["BFGS", "L-BFGS-B"]:
            if iter_n == 0:
                log_opt(0)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                training_loss,
                m.trainable_variables,
                callback=log_opt,
                options={"maxiter": iterations},
                method=optimiser,
            )

        else:
            assert learning_rate is not None

            if optimiser=="Adam":
                opt = tf.optimizers.Adam(learning_rate)
            elif optimiser=="SGD":
                opt = tf.optimizers.SGD(learning_rate)

            @tf.function
            def optimisation_step():
                opt.minimize(
                    training_loss,
                    m.trainable_variables,
                    )

            #* Carry out optimsation iterations
            freq = max(iterations//n_logs, 1)
            for i in tqdm(range(iterations)):
                if i % freq == 0:  #only log every freq iterations
                    log_opt(i)
                optimisation_step()
                if i == iterations - 1:
                    log_opt(i+1)
    
    def reduce_lengthscales(m):
        print("reducing lengthscales slightly...\n", "old lengthlscales", m.kernel.lengthscales.numpy())
        new_l = m.kernel.lengthscales.numpy()
        mean = np.mean(new_l[new_l > 0])
        new_l[new_l == 0] = mean
        new_l *= 0.9
        m.kernel.lengthscales.assign(new_l)
    

    
    #- Optimisation process
    
    #* Print state of model before optimisation
    gpflow.utilities.print_summary(m)  

    # //raise Exception("Finishing code here")  #%% Finishes here    

    #* Print starting message
    print("Starting optimisation for SGPR model with {} inducing points. \n Notes: {}".format( num_inducing_points, notes))

    num_cholesky_errors = 0
    num_extra_reinits = 0

    while num_cholesky_errors < 10:
        try:
            #* Optimise (different process for greedy var - reinit)
            if opt_strategy=="reinit" or opt_strategy=="reinittrain":

                gpflow.set_trainable(m.inducing_variable, False)

                assert initial_inducing_point_locations=="Greedy Variance"
                for i in range(num_iterations//reinit_freq):
                    print("Initialisation {}:".format(i))
                    run_optimizer(
                        learning_rate=opt_learning_rate,
                        iterations=reinit_freq,
                        )
                    old_Z = m.inducing_variable.Z.numpy().copy()
                    old_elbo = m.maximum_log_likelihood_objective()
                    Z_initer = ConditionalVariance()
                    
                    Z = Z_initer.compute_initialisation(
                        x_train,
                        num_inducing_points,
                        kernel_obj)[0]
                    m.inducing_variable.Z.assign(Z)

                    if m.maximum_log_likelihood_objective() <= old_elbo:

                        if opt_strategy=="reinit":
                            if num_extra_reinits < 10:
                                print("Carrying out reinitialisation even though old elbo is better. Number of extra reinits: {}/10".format(num_extra_reinits))
                                num_extra_reinits += 1
                                continue
                            else:
                                # Restore old Z, and finish optimisation
                                m.inducing_variable.Z.assign(old_Z)
                                print("Stopped reinit_Z procedure after {} initialisations because new ELBO was smaller than old ELBO. Training for 100 more iterations to ensure convergence.".format(i))
                                run_optimizer(
                                    learning_rate=opt_learning_rate,
                                    iterations=200,
                                    )
                                break
                        
                        if opt_strategy=="reinittrain":
                            print("Training with reinitialised inducing variables done (new elbo was worse than old elbo). Setting inducing variables as trainable and training again...")
                            m.inducing_variable.Z.assign(old_Z)
                            gpflow.set_trainable(m.inducing_variable, True)
                            run_optimizer(opt_learning_rate, num_iterations-iter_n)
                            break

                break
                        
            elif opt_strategy=="retrain":
                gpflow.set_trainable(m.inducing_variable, False)
                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                print("Training with fixed inducing variables done. Reinitialising Z, setting inducing variables as trainable and training again...")
                Z_initer = ConditionalVariance()
                    
                Z = Z_initer.compute_initialisation(
                    x_train,
                    num_inducing_points,
                    kernel_obj)[0]
                m.inducing_variable.Z.assign(Z)
                gpflow.set_trainable(m.inducing_variable, True)

                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                break
            elif opt_strategy=="train":
                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                break
            elif opt_strategy=="init":
                gpflow.set_trainable(m.inducing_variable, False)
                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                break
            else:
                raise ValueError("Something went wrong when implementing optimisation strategy. Something needs fixing.")
        
        except KeyboardInterrupt:
            print("Training stopped early. Logging model anyway...")
            break
        except (InvalidArgumentError, FloatingPointError):
            num_cholesky_errors += 1
            # reduce_lengthscales(m)
            if 7 > num_cholesky_errors > 2: #* we raise Jitter
                print("Cholesky error happened again (total of {} times now), raising jitter from {} to sqrt 10 times bigger".format(num_cholesky_errors, jitter))
                jitter *= 10**0.5 # multiply by sqrt 10
                gpflow.config.set_default_jitter(jitter); #adaptive jitter?
            elif num_cholesky_errors > 6:
                print("so many errors now:", num_cholesky_errors, " in total. Moving on...")
                break
            else:   #* We initialise with new Z
                print("Cholesky error occurred (total errors = {})...reinitialising Z with greedy variance".format(num_cholesky_errors))
                Z_initer = ConditionalVariance()
                print("Updating model with new Z...")
                Z = Z_initer.compute_initialisation(
                    x_train,
                    num_inducing_points,
                    kernel_obj)[0]
                m.inducing_variable.Z.assign(Z)
                print("Trying again with new Z!")
                gpflow.utilities.print_summary(m) 


    #* Stop Stopwatch for the final time
    watch.stop()
    
    #* Print results to the console
    print("Model: SGPR, Kernel: {}, Dataset: {}".format(kernel, dataset))
    print("Number of inducing points: {}".format(num_inducing_points))
    print("Total time for optimisation: {} seconds".format(watch.read_time()))
    print("Notes:", notes)
    gpflow.utilities.print_summary(m)




    #- Gather final model hyperparameters

    final_hyperparams = {param[1:].replace(".", " "): value.numpy() for (param, value) in gpflow.utilities.parameter_dict(m).items()}

    initial_hyperparams = hyperparams_dict





    return (
        dataset,
        "SGPR",
        num_inducing_points,
        opt_strategy,
        optimiser,
        opt_learning_rate,
        "None",
        num_iterations,
        times,
        elbo_logs,
        rmse_logs,
        nlpd_logs,
        "None",
        initial_hyperparams,
        final_hyperparams,
        m,
        notes
        )


def full_gp_trainer(
    dataset,
    num_iterations,
    optimiser="BFGS",
    opt_learning_rate=None,
    fixed_parameters=None,
    *,
    kernel = "ARD SE",
    initial_kernel_variance=None,
    initial_kernel_lengthscale=None,
    initial_likelihood_hyperparameter=None,
    notes=None,
    parameter_logging=None,
    output_plot_csv=False,
    normalise_dataset=True,

    ):


    model = "Full GP"


    #- Start stopwatch to time setting up the model
    
    #* Stopwatch instance
    watch = Stopwatch()
    watch.reset()
    #* Start the Stopwatch
    watch.start()


    #- Set up data

    #* Load dataset into 4 arrays
    x_train, x_test, y_train, y_test = load_dataset(
        dataset=dataset,
        y_hot_encoding = False,
        normalised=normalise_dataset,
        )

    #* Convert training and test datasets into Tensorflow dataset objects
    original_training_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
        );
    original_test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
        );




    
    #- Configure settings for model

    num_training_datapoints = len(y_train)
    num_dim = len(x_train[0])

    #* Number of logs to make during optimisation
    n_logs = 1000

    #* Pick value for jitter
    jitter = 1e-5 # 1e-4
    gpflow.config.set_default_jitter(jitter); #adaptive jitter?
    # gpflow.config.set_default_positive_minimum(1.0e-5)


    #* Kernel
    if kernel in {"SE", "ARD SE"}:
        if kernel == "SE":
            kernel_obj = gpflow.kernels.SquaredExponential()
        elif kernel == "ARD SE":
            kernel_obj = gpflow.kernels.SquaredExponential(lengthscales=[1.]*num_dim)
    else:
        raise ValueError("Kernel not supported.")







    #- Set up the model
   
    
    m = gpflow.models.GPR(
        data=(x_train, y_train),
        kernel=kernel_obj,
        )


    
    #* Initialise hyperparams and set trainable ones.
    def initialise_hyperparameters(initial_kernel_lengthscale, initial_kernel_variance, initial_likelihood_hyperparameter):
        #* Initialise hyperparams:
        # # This is done using heuristics from https://infallible-thompson-49de36.netlify.app/

        if initial_kernel_variance is None:
            initial_kernel_variance = np.std(y_train)
        m.kernel.variance.assign(initial_kernel_variance)

        if initial_kernel_lengthscale is None:
            if kernel=="SE":
                initial_kernel_lengthscale = np.std(x_train)
            if kernel=="ARD SE":
                initial_kernel_lengthscale = np.std(x_train, axis=0)
                mean = np.mean(initial_kernel_lengthscale[initial_kernel_lengthscale > 0])
                initial_kernel_lengthscale[initial_kernel_lengthscale == 0] = mean
            print("initial kernel lengthscale not chosen. Using: ", initial_kernel_lengthscale)
        m.kernel.lengthscales.assign(initial_kernel_lengthscale)

        if initial_likelihood_hyperparameter is None:
            initial_likelihood_hyperparameter = initial_kernel_variance/50
        m.likelihood.variance.assign(initial_likelihood_hyperparameter)


        #* Constrain hyperparameters such that they do not go to wacky values during optimisation:
        lengthscale_constrained_transform = tfp.bijectors.Sigmoid(
            gpflow.utilities.to_default_float(min(m.kernel.lengthscales.numpy())/100),
            gpflow.utilities.to_default_float(max(m.kernel.lengthscales.numpy())*100),
        )

        var_constrained_transform = tfp.bijectors.Sigmoid(
            gpflow.utilities.to_default_float(m.kernel.variance.numpy()/1000),
            gpflow.utilities.to_default_float(m.kernel.variance.numpy()*1000),
        )

        noise_constrained_transform = tfp.bijectors.Sigmoid(
            gpflow.utilities.to_default_float(jitter*10), # minimum the noise can be is 10 times jitter.
            gpflow.utilities.to_default_float(m.likelihood.variance.numpy()), # max is the kernel variance
        )

        print(gpflow.utilities.to_default_float(m.likelihood.variance.numpy()),jitter*10)

        new_len = gpflow.Parameter(
            m.kernel.lengthscales.numpy(),transform=lengthscale_constrained_transform)
        new_var = gpflow.Parameter(
            m.kernel.variance.numpy(),
            transform = var_constrained_transform)
        # new_noi = gpflow.Parameter(
        #     m.likelihood.variance.numpy(),
        #     transform = noise_constrained_transform)
        
        m.kernel.lengthscales = new_len
        m.kernel.variance = new_var
        # m.likelihood.variance = new_noi

    def set_trainable_parameters(parameters_set):
        if "likelihood variance" in parameters_set:
            gpflow.set_trainable(m.likelihood.variance, False)
        if "kernel variance" in parameters_set:
            gpflow.set_trainable(m.kernel.variance, False)
        if "kernel lengthscales" in parameters_set:
            gpflow.set_trainable(m.kernel.lengthscales, False)
    
    initialise_hyperparameters(
        initial_kernel_lengthscale,
        initial_kernel_variance,
        initial_likelihood_hyperparameter
    )
    if fixed_parameters is not None:
        set_trainable_parameters(fixed_parameters)

    
    
    #* Define Minibatch dataset and training loss
    training_loss=m.training_loss


   
    
    #- Set up necessary instances and functions for training

    if output_plot_csv:
        if notes is not None:
            notes_ = notes.replace(" ", "_")
        else:
            notes_ = ""
        temp_filename = "temp_metric_log_[{}].csv".format(notes_)
        cols = ["time", "ELBO", "RMSE", "NLPD"]

        #* Create an empty dataframe with columns
        log_df  = pd.DataFrame(columns = cols)

        #* Save the empty df into a csv
        log_df.to_csv(
            temp_filename,
        )
        def log_into_csv(time, elbo, rmse, nlpd):
            log_df.loc[len(log_df.index)] = (time, elbo, rmse, nlpd)
            log_df.iloc[[-1]].to_csv(
                temp_filename,
                mode="a",
                header=False,
            )

            
        

    #* Counter for BFGS optimisation logging
    global iter_n
    iter_n = 0

    #* List for ELBO values during 
    elbo_logs = []
    rmse_logs = []
    nlpd_logs = []
    parameter_logs = []
    times = [] 

    def log_rmse(Y_predict_mu, Y_test):
        """Root mean squared error loss

        Args:
            Y_predict_mu
            Y_test

        Returns:
            rmse
        """
        rmse = np.sqrt(np.mean((Y_predict_mu - Y_test)**2))
        rmse_logs.append(rmse)
        return rmse

    def log_nlpd(Y_predict_mu, Y_predict_var, Y_test, X_test):
        """Loss function - negative log validation density loss.

        Args:
            Y_predict_mu (_type_): 
            Y_predict_var (_type_): 
            Y_test (_type_): 

        """

        term_1 = np.log(Y_predict_var)
        term_2 = ((Y_predict_mu - Y_test)**2)/(Y_predict_var)
        term_3 = np.log(2*np.pi)
        nlpd = np.sum(-0.5*(term_1+term_2+term_3))
        nlpd_per_dp = nlpd/len(Y_test)
        nlpp = -np.mean(m.predict_log_density((X_test, Y_test)))

        nlpd_logs.append(nlpp)

        return nlpp

    
    def log_parameters():
        """Logs a dictionary of the parameters into a list.
        Args:
            parameters (string): the parameters to be logged.
        """
        if parameter_logging=="all":
                parameter_logs.append({"kernel variance": m.kernel.variance.numpy().tolist(), "likelihood variance": m.likelihood.variance.numpy().tolist(), "kernel lengthscales": m.kernel.lengthscales.numpy().tolist()})

    def log_opt(iteration):
        """
        Utility function to log the elbo score for each iteration during optimisaiton.

        Args:
            iteration number (just for printing)
        """
        #* Stop the watch before logging anything
        watch.stop()
        global iter_n

        #* Calculate elbo
        
        elbo =  m.maximum_log_likelihood_objective()
        
        

        #* Make predictions
        y_pred_mu, y_pred_var = m.predict_y(x_test)
        
        #* Log results
        nlpd_loss = log_nlpd(y_pred_mu, y_pred_var, y_test, x_test)
        rmse_loss = log_rmse(y_pred_mu, y_test)
        elbo_logs.append(float(elbo))
        times.append(watch.read_time())
        if parameter_logging is not None:
            log_parameters(parameter_logging)

        #* Print optimisation stats to console
        if optimiser=="Adam" or optimiser=="SGD":
            tqdm.write("Iteration: {}, RMSE: {:.4e}, NLPD: {:.4f}, ELBO: {:.4e}".format(iteration, rmse_loss, nlpd_loss, elbo))
        elif optimiser in {"BFGS", "L-BFGS-B"}:
            print("Iteration: {}, RMSE: {:.4e}, NLPD: {:.4f}, ELBO: {:.4e}".format(iter_n, rmse_loss, nlpd_loss, elbo))

        if output_plot_csv:
            log_into_csv(watch.read_time(), float(elbo), rmse_loss, nlpd_loss)

        #* Increment iteration counter
        iter_n += 1 

        #* Start stopwatch again
        watch.start()


    def run_optimizer(learning_rate, iterations):
        """Carries out optimisation of model's objective function

        Args:
            learning_rate (float)
            iterations (integer)
        """
        global iter_n
        #* Optimiser object
        if optimiser in {"BFGS", "L-BFGS-B"}:
            if iter_n == 0:
                log_opt(0)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                training_loss,
                m.trainable_variables,
                callback=log_opt,
                options={"maxiter": iterations},
                method=optimiser,
            )

        else:
            if learning_rate is None:
                learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                    0.1,
                    decay_steps=num_iterations,
                    decay_rate=0.96,
                    staircase=True
                )
            if optimiser=="Adam":
                opt = tf.optimizers.Adam(learning_rate)
            elif optimiser=="SGD":
                opt = tf.optimizers.SGD(learning_rate)

            @tf.function
            def optimisation_step():
                opt.minimize(
                    training_loss,
                    m.trainable_variables,
                    )

            #* Carry out optimsation iterations
            freq = max(iterations//n_logs, 1)
            for i in tqdm(range(iterations)):
                if i % freq == 0:  #only log every freq iterations
                    log_opt(i)
                optimisation_step()
                if i == iterations - 1:
                    log_opt(i+1)
    
    def reduce_lengthscales(m):
        print("reducing lengthscales slightly...\n", "old lengthlscales", m.kernel.lengthscales.numpy())
        new_l = m.kernel.lengthscales.numpy()
        mean = np.mean(new_l[new_l > 0])
        new_l[new_l == 0] = mean
        new_l *= 0.9
        m.kernel.lengthscales.assign(new_l)
    

    
    #- Optimisation process
    
    #* Print state of model before optimisation
    gpflow.utilities.print_summary(m)        

    #* Print starting message
    print("Starting optimisation for {} model. \n Notes: {}".format(model, notes))

    num_cholesky_errors = 0
    num_extra_reinits = 0

    while num_cholesky_errors < 10:
        try:
            #* Optimise (different process for greedy var - reinit)
                run_optimizer(opt_learning_rate, num_iterations-iter_n)
                break
        
        except KeyboardInterrupt:
            print("Training stopped early. Logging model anyway...")
            break
        except (InvalidArgumentError, FloatingPointError):
            num_cholesky_errors += 1
            # reduce_lengthscales(m)
            if num_cholesky_errors < 7: #* we raise Jitter
                print("Cholesky error happened again (total of {} times now), raising jitter from {} to sqrt 10 times bigger".format(num_cholesky_errors, jitter))
                jitter *= 10**0.5 # multiply by sqrt 10
                gpflow.config.set_default_jitter(jitter); #adaptive jitter?
            elif num_cholesky_errors > 6:
                print("so many errors now:", num_cholesky_errors, " in total. Moving on...")
                break
            

    #* Stop Stopwatch for the final time
    watch.stop()
    
    #* Print results to the console
    print("Model: {}".format(model))
    print("Total time for optimisation: {} seconds".format(watch.read_time()))
    print("Notes:", notes)
    gpflow.utilities.print_summary(m)




    #- Gather final model hyperparameters


    final_hyperparams = {
        "kernel variance": m.kernel.variance.numpy().tolist(),
        "kernel lengthscale": m.kernel.lengthscales.numpy().tolist(),
        "likelihood variance": m.likelihood.variance.numpy().tolist(),
    }

    initial_hyperparams = {
        "kernel variance": initial_kernel_variance,
        "kernel lengthscale": initial_kernel_lengthscale,
        "likelihood variance": initial_likelihood_hyperparameter,
    }





    return (
        dataset,
        "Full GP",
        "N/A",
        fixed_parameters,
        optimiser,
        opt_learning_rate,
        "N/A",
        num_iterations,
        times,
        elbo_logs,
        rmse_logs,
        nlpd_logs,
        parameter_logs,
        initial_hyperparams,
        final_hyperparams,
        m,
        notes
        )




if __name__ == "__main__":
    
    import pandas as pd
    results = pd.Series(
        full_gp_trainer(
            "wilson_energy",
            2000,
            "L-BFGS-B",
            normalise_dataset = False,


        )
    )
    # print("Final ELBO", results[15].elbo().numpy())

    while True:
        save_results = input("Save results? (y/n) ")
        if save_results in {"y","n"}:
            break
        else:
            print("Please enter a valid answer")
    
    if save_results == "y":
        filename = input("Enter the filename pls: ")
        
        
    plt.figure()
    plt.plot(results[8], results[9])
    plt.xlabel("Time (s)")
    plt.ylabel("ELBO")
    plt.show()
    
    # rmse_lin_3_spline = lin_mod_scores("1D", "spline", 3)
    # plt.figure()
    # plt.plot(results[8], results[10])
    # plt.hlines(rmse_lin_3_spline, 0, np.max(results[8]), label="3rd order spline", alpha=0.3)
    # plt.xlabel("Time (s)")
    # plt.ylabel("RMSE")
    # plt.show()

    # plt.figure()
    # plt.plot(results[8], results[11])
    # plt.xlabel("Time (s)")
    # plt.ylabel("NLPD")
    # plt.show()


    
# %%
