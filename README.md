# Bachlor thesis: A Comparison between Pruning Techniques and their Applicability to the Lottery Ticket Hypothesis
#### by Felix Japtok
## Describtion
This is the code I used for the experiment conducted in my thesis "A Comparison between Pruning Techniques and their Applicability to the Lottery Ticket Hypothesis".
All saved models from the experiment are included in the "saved_models" directory and can be loaded by editing the variable "args.resume" in "experiment.py".
The collected data and metrics for the models are located in "stats", all loss and accuracy values for all models were combined into "full_loss_accuracy" for easier plotting.
All plots included in the thesis with a few additonal ones can be found in "graphics". All of them were created with slight variations of the functions in "plot.py".
## Installation
1. download or clone this repository
2. install anaconda or something comparable on your device
3. navigate to the repository directory and open the terminal
4. create the environment to run the experiment
 - conda env create -f environment.yml (with conda)
5. activate the created enrivonment "comparison_exp"
 - conda activate comparison_exp (with conda)
6. execute the experiment.py file
7. execute plot.py to generate some of the utilized plots in the thesis

Alternatively it is also possible to install the dependencies in "environment.yml" manually and skip to 6. of installation
