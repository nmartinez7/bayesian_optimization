# Bayesian Optimization Exercise - Oil Exploration
Bayesian optimization is an amazing tool for solving problems where getting answers is expensive, slow, or dangerous.
My first exposure to Bayesian optimization was for solving hyperparameter optimization of deep neural networks. It was common
to start with a simple grid search or random search, but that simply gets out of hand once the model and or dataset size 
are large enough. Bayesian optimization tools like scikit-optimize or Optuna help to minimize the time spent on hyperparameter
tuning by being "smart" about it and working like a treasure hunter: using every peace of information from past experiments
to decide exactly where to look next.
