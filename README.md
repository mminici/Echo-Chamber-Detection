Before running any experiment, we suggest you install the provided conda environment.
> How to install the conda environment: ```conda env create --file environment.yml```

# Running an experiment
In both sections, the parameters highlighted in bold are the ones you should modify in order to fully reproduce our results.

## Synthetic Experiments

In order to run a synthetic experiment, you can use ```python grid-synthetic.py``` if you want to use default settings, otherwise, you can specify several options:
* **seed**, to change the initial random seed for stochasticity;
* N, to change the number of initial users;
* eta, to change the polarity of each community;
* **s**, to change the social prior size;
* **h**, to change the echo-chamber prior size;
* B, you can ignore this parameter;
* **items_per_node**, to change the number of items per user;
* lr, learning rate of ECD procedure;
* epochs, number of epochs of ECD procedure;
* oversampling, whether you want to oversample the minority class --- either edges or propagations;
* reweighting, whether you want to reweight the two classes w.r.t. their cardinality;
* annealing, whether you want to add an annealing procedure;
* ablation, whether you want to exclude links or propagations from the ECD procedure;
* training_type, you can ignore this parameter;
* model_type, you can ignore this parameter;
* device, whether you want the experiment to be run on GPU. If no GPU is available, then pass an empty string.

## Real-World Experiments

In order to run a synthetic experiment, you can use ```python grid-real.py``` if you want to use default settings, otherwise, you can specify several options:
* **seed**, to change the initial random seed for stochasticity;
* **dataset**, to change the dataset to analyze. Pick one between ```brexit``` and ```vaxNoVax```.
* **K**, to change the number of communities you want to find;
* **s**, to change the social prior size;
* **h**, to change the echo-chamber prior size;
* * B, you can ignore this parameter;
* lr, learning rate of ECD procedure;
* epochs, number of epochs of ECD procedure;
* oversampling, whether you want to oversample the minority class --- either edges or propagations;
* reweighting, whether you want to reweight the two classes w.r.t. their cardinality;
* annealing, whether you want to add an annealing procedure;
* link_removal, you can ignore this parameter!
* **prop_removal**, whether you want to exclude some propagations from the dataset;
* **prop_removal_perc**, the percentage of propagations you want to remove;
* **stance_detection_exp**, it is a boolean flag --- i.e.: True or False --- that when you want to analyze the stance detection capability of ECD if set to True, removes all propagations of the analyzed users.
* ablation, whether you want to exclude links or propagations from the ECD procedure;
* training_type, you can ignore this parameter;
* model_type, you can ignore this parameter;
* device, whether you want the experiment to be run on GPU. If no GPU is available, then pass an empty string.

