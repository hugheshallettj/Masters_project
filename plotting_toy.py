# from this import d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ast import literal_eval
from gp_trainers import lin_mod_scores
import glob
from matplotlib.offsetbox import AnchoredText

import regex as re

from gp_trainers import load_dataset
import gp_trainers


col_names =  [
        "Dataset",
        "Model",
        "M",
        "Fixed parameters",
        "Optimiser",
        "Optimiser learning rate",
        "Minibatch size",
        "Iterations",
        "Time logs",
        "ELBO logs",
        "RMSE logs",
        "NLPD loss logs",
        "Parameter logs",
        "Initial hyperparams",
        "Final hyperparams",
        "GPflow model object",
        "Notes"
        ]

results_df  = pd.DataFrame(columns = col_names)

include_rbf = True
include_order_2 = True
include_cubic_arc = True

kernel_titles=[]


if include_rbf:
    results_df.loc[len(results_df.index)] = gp_trainers.gp_regression_trainer(
                dataset="1D_exp+cubic+step",
                model="SGPR",
                num_inducing_points=50,
                initial_inducing_point_locations="Greedy Variance",
                num_iterations=500,
                optimiser="L-BFGS-B",
                opt_strategy="reinittrain",
                kernel="ARD SE"
                )
    kernel_titles.append("Squared Exponential")

if include_order_2:
    results_df.loc[len(results_df.index)] = gp_trainers.gp_regression_trainer(
            dataset="1D_exp+cubic+step",
            model="SGPR",
            num_inducing_points=50,
            initial_inducing_point_locations="Greedy Variance",
            num_iterations=500,
            optimiser="L-BFGS-B",
            opt_strategy="reinittrain",
            kernel="ARD ArcCosine (order 2)"
            )
    kernel_titles.append("ArcCosine (order 2)")

results_df.loc[len(results_df.index)] = gp_trainers.gp_regression_trainer(
            dataset="1D_exp+cubic+step",
            model="SGPR",
            num_inducing_points=50,
            initial_inducing_point_locations="Greedy Variance",
            num_iterations=500,
            optimiser="L-BFGS-B",
            opt_strategy="reinittrain",
            kernel="ARD ArcCosine (order 1)"
            )
kernel_titles.append("ArcCosine (order 1)")

results_df.loc[len(results_df.index)] = gp_trainers.gp_regression_trainer(
            dataset="1D_exp+cubic+step",
            model="SGPR",
            num_inducing_points=50,
            initial_inducing_point_locations="Greedy Variance",
            num_iterations=500,
            optimiser="L-BFGS-B",
            opt_strategy="reinittrain",
            kernel="ARD ArcCosine"
            )
kernel_titles.append("ArcCosine (order 0)")

if include_cubic_arc:
    results_df.loc[len(results_df.index)] = gp_trainers.gp_regression_trainer(
            dataset="1D_exp+cubic+step",
            model="SGPR",
            num_inducing_points=50,
            initial_inducing_point_locations="Greedy Variance",
            num_iterations=500,
            optimiser="L-BFGS-B",
            opt_strategy="reinittrain",
            kernel="ARD ArcCosine + Polynomial (3)"
            )
    kernel_titles.append("ArcCosine (0) + Polynomial (3)")







x_train, x_test, y_train, y_test = load_dataset("1D_exp+cubic+step")
X_predict = np.linspace(-0.035, 0.035, 300)


fig, axs = plt.subplots(len(results_df), 1, figsize=(7,4*len(results_df)), tight_layout=True)
for i in range(len(results_df)):
    print("plotting for: ", kernel_titles[i])
    m = results_df["GPflow model object"][i]
    Z = m.inducing_variable.Z.numpy()
    try:
        elbo = m.elbo().numpy()
    except gp_trainers.InvalidArgumentError:
        elbo = "N/A"
    y_predict, var = m.predict_y(X_predict.reshape(-1, 1), full_cov=False)


    y_predict, var = np.array(y_predict).flatten(), np.array(var).flatten()


    axs[i].set_title(kernel_titles[i])
    axs[i].scatter(x_train, y_train, color = "red", marker=".", s=0.3 )

    p3 = axs[i].scatter(Z, np.zeros(np.shape(Z)), marker="|", color="black", s=7)
    p1, = axs[i].plot(X_predict, y_predict, color="green")
    axs[i].fill_between(X_predict, y_predict+np.sqrt(var)*1.96, y_predict-np.sqrt(var)*1.96, alpha=0.2, color = "green")
    axs[i].grid(alpha=0.5)
    axs[i].set_ylim(-3.5, 3.2)
    at = AnchoredText(
        "ELBO: {:.4e}".format(elbo), prop=dict(size=12), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[i].add_artist(at)
    axs[i].set_ylabel(r"$y$")




axs[len(results_df)-1].set_xlabel(r"$x$")
axs[1].set_ylabel(r"$y$")



p2 = mpatches.Patch(color='g', alpha=0.2, linewidth=0)
fig.legend([(p2, p1), p3], ['Prediction', 'Inducing Point Inputs'],)

filename="toy_demo.pdf"
# plt.savefig(f"/Users/jacob/Downloads/{filename}")
plt.show()
