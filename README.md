# Bayesian Optimization Exercise - Oil Exploration
Bayesian optimization is an amazing tool for solving problems where getting answers is **expensive, slow, or dangerous**.
During my career as a data scientist, I was first exposed to Bayesian optimization was for solving hyperparameter optimization of deep neural networks. It was common
to start with a simple grid search or random search, but that simply gets out of hand once the model and or dataset size 
are large enough. Bayesian optimization tools like scikit-optimize or Optuna help to minimize the time spent on hyperparameter
tuning by being "smart" about the process and working like a treasure hunter: using every peace of information from past experiments
to decide exactly where to look next.
 
Bayesian optimization has applications in many fields, including:

* Drug discovery: a drug trial can take months and cost hundreds of thousands of dollars. Bayesian optimization (BO) can help to navigate
the search space and pick the next dosage/concentration/etc. wisely.
* Robotics and autonomous systems: testing physicals robots is risky and expensive. Figuring out the best walking gait for a 
bipedal robot using random search might result in the robot falling over and breaking down. Using BO allows engineers to tune controllers
or reinforcement learning policies with only small numbers of physical trials.
* A/B testing and marketing: if you are testing 20 different versions of an ad, you don't want to waste traffic on the low-performing ones for long.
BO allows marketing teams to shift toward the best-performing versions of the website in real-time.

I could go on, with use cases but I'll finally talk about the one in this example: oil exploration.