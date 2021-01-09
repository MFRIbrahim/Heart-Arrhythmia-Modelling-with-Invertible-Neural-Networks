# Heart Arrhythmia Modelling with Invertible Neural-Networks
This repository contains the code for my master thesis "Heart Arrhythmia Modelling with Invertible Neural Networks" (Ruprecht Karl University of Heidelberg, Institute for Computer Science, Visual Learning Lab). It includes the code used for data generation, model creation as well as training and testing the models.
The code for the forward model is provided on https://github.com/mathemedical/HEAT_forwardSim and is required for both "datagen.py" and "training.py" of the code on this repository.

If you want to try out the models with arbitrary RR-intervals use the function `get_solution()` in "training.py". It takes an RR-interval list and a network name as inputs. The network names are the same ones used in the thesis to describe the different approaches: "bp_cINN", "bp_cINN_multi", "bp_rcINN", "bp_rcINN_matching", "signal_cINN", "signal_rcINN", "signal_rcINN_matching". The function writes the top 10 suggested forward model parameters as well as the resulting RR-interval lists to a file "top10_sol_{name}.txt", where "name" is the network name mentioned above. The following format is used for the file:

````
--------------------------
[680.98599243 453.99066162 453.99066162 680.98599243 453.99066162]
9.5887939453125
[[1, 2, 1, 1, 2, 1], [1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
2c
[226.99533081054688, 174.15582275390625]
--------------------------


================================================
[669. 446. 446. 669. 446.]
================================================
````

Surrounded by two dashed lines is the resulting RR-interval list when using the suggested forward model parameters by the network, the MAE between the resulting RR-interval list and the given RR-interval list, the suggested block pattern (all three levels are always shown), the block type, the atrial cycle length and the conduction constant. There can be 0-10 of such dashed line blocks depending on how many valid solutions the network suggests. After that, surrounded by equal signs, is the given RR-interval list. Additionally, for the "signal" approaches the signal matrix is also given at the end of a dashed line block. In case you want to try a lot of RR-intervals at once consider using the function `get_solution_mp()`. This function can take multiple RR-interval lists given in a list as well as one network name and uses multiprocessing to speed up the computation.
