# Specifications of Human-Robot Encounters: Learning STL formulae in a Multi-Label Multi-Class Approach
Dataset and code to formalize trajectories in human-robot encounters, where trajectories can present multiple labels


## Will be moved to planiacs repo 21.02!


## Introduction

We are interested in formalizing human trajectories in human-robot encounters.
Typically, STL inference methods learn from data partitioned between negative and positive instances.
However, these methods do not account for cases where the positive data can contain several classes of pre-identified behaviors.
We propose a decision tree-based algorithm to extract STL formulae from multi-labelled data.
We apply our method to a dataset of trajectories collected through an online study where participants had to avoid colliding with a robot in a shared environment. The human participants described different behaviors, ranging from being in a hurry/minimizing completion time to maximizing safety.


## Downloading sources

You can use this API by cloning this repository:
```
$ git clone https://github.com/allinard/stl_multiclass
```

Dependencies:
* Python 3.8
	* numpy
	* matplotlib
	* scipy
	* sklearn
	* dill
	* pandas
	* Pulp
* Gurobi MILP Solver




## Inference of multiclass STL formulae

Will be posted soon.




## Experiments

You will find here 2 experiments we ran on 2 different datasets.

### Synthetic dataset

In this experiment, we generated 500 trajectories given generated STL specifications. We considered

![Alt text](experiments_img/synthetic_data_specs.PNG)

and the following classes of trajectories:

![Alt text](experiments_img/synthetic_data_classes.PNG)

Since we consider the multi-class and multi-label case, trajectories could be labelled as {c1}, {c2}, {c3}, {c1, c2}, {c1, c3}, {c2, c3} and {c1, c2, c3}. We generated trajectories from specifications of different (combinations of) classes using an MILP approach and the Gurobi optimizer.
We generated 100 trajectories for each of the classes {c1}, {c2} and {c3}, and 50 trajectories for each of the remaining classes {c1, c2}, {c1, c3}, {c2, c3} and {c1, c2, c3}. Here are example trajectories for each of the classes:

![Alt text](experiments_img/synthetic_data_map.PNG)

We evaluated our methods over these classes by cross-validation (on 5 folds), for of our baseline (dt), and STL-difference (dtΔ) methods compared to a classical neural networks approach (nn), where H represents the results in terms of hamming loss, and A the results in terms of example accuracy:

![Alt text](experiments_img/synthetic_data_res_cv.PNG)

Finally, we could learn the following models:

![Alt text](experiments_img/synthetic_data_models.png)



### User study data on human-robot encounters

We ran experiments on a dataset of trajectories collected through an online study where participants had to avoid colliding with a robot in a shared environment, and where the participants depicted 3 behaviors: being in a hurry, taking a normal walk, or maximizing safety.

![Alt text](experiments_img/userstudy_data_screenshot.PNG)

We collected a total of 900 trajectories (50 participants x 6 trials x 3 motivations). After filtering out outliers, we used 842 trajectories.
Since participants may have different conceptions and descriptions of the different behaviors, we notice some overlaps in the trajectories they depict. Indeed, some participant's trajectories under the "carrying something fragile" mode might overlap the behavior of participants when they are “taking a normal walk”. Therefore, we want to associate these multiple labels to such trajectories.
The dataset is composed as follows (where 'f' is the "safety first" motivation, 'w' is the "taking a walk" motivation and 'h' is the "being in a hurry" motivation):

![Alt text](experiments_img/userstudy_data_datacomposition.PNG)

We evaluated our methods over the user study data by cross-validation (on 5 folds), for of our baseline (dt), and STL-difference (dtΔ) methods compared to a classical neural networks approach (nn), where H represents the results in terms of hamming loss, and A the results in terms of example accuracy. The maximum height of the decision trees was set to 5:

![Alt text](experiments_img/userstudy_data_res_cvh5.PNG)

Here the results obtained with a maximum height of the decision trees set to 10:

![Alt text](experiments_img/userstudy_data_res_cvh10.PNG)

On the whole dataset, we could learn the following models. Results of the classification of the signals are shown by the confusion matrix below the models:

![Alt text](experiments_img/userstudy_data_models.png)

![Alt text](experiments_img/userstudy_data_cm.PNG)


## Publications

You can find hereinafter the related publication f<or more details on the implemented methods:
* TBD/Submitted
