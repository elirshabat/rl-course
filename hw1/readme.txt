------------------------------
Names
------------------------------
Alon Ressler, 201547510, alonress@gmail.com 
and 
Eliran Shabat, 201602877, shabat.eliran@gmail.com

------------------------------
Execution Directory
------------------------------
/specific/a/home/cc/students/cs/eliranshabat/courses/msc/RL/hw1

------------------------------
Q1 Inputs Description
------------------------------
There are 4 source files.

1. mnist.py
Runs the code that correspond to part 1 of the quesion.
It runs 100 epochs with hyper parameters as they are defined in the template file.

2. mnist_optimized.py
Runs the code that correspond to part 2 of the quesion.
It runs 100 epoches with optimized hyper paramters and the same network as in part 1.
The optimized parameters are the learning-rate (0.01 instead of 0.001) and the batch-size (50 instead of 100).

3. mnist_deep.py
Runs the code that correspond to part 3 of the quesion.
It runs 100 epoches using a deep network with one hidden layer.

4. mnist_report.py
Generate plots (images) and prints the accuracy of each part of the quesion.


------------------------------
Q1 Commands
------------------------------
> /usr/local/lib/anaconda3-5.1.0/bin/python mnist_deep.py mnist.py
Runs part 1 of the quesion.

> /usr/local/lib/anaconda3-5.1.0/bin/python mnist_deep.py mnist_optimized.py
Runs part 2 of the quesion.

> /usr/local/lib/anaconda3-5.1.0/bin/python mnist_deep.py mnist_deep.py
Runs part 3 of the quesion.

> /usr/local/lib/anaconda3-5.1.0/bin/python mnist_report.py
Runs the code that generate plots and print the accuracy.
Note: it assumes that the previous command have already run (they generate data files needed for this module to run).


------------------------------
Q1 Output Description
------------------------------
Data files (located in ./out):
1. mnist_output_data.json - json file with configuration and loss information of part 1 of the quesion.
2. mnist_optimized_output_data.json - json file with configuration and loss information of part 1 of the quesion.
3. mnist_deep_output_data.json - json file with configuration and loss information of part 1 of the quesion.

Plot files (located in out/figures):
1. loss_curve_normal.png - plot the loss on each bath of part 1 of the quesion.
2. loss_curve_optimized.png - plot the loss on each bath of part 2 of the quesion.
3. loss_curve_deep.png - plot the loss on each bath of part 3 of the quesion.
4. average_loss_curve.png - plot the average loss on each epoch for each of the 3 parts (for comparison).

Model parameter files (located in out/model):
1. normal_model.pkl - model parameters of the un-optimized code.
2. optimized_model.pkl - model parameters of the optimized code.
3. deep_model.pkl - model parameters of the deep network.


------------------------------
Q2 Inputs Description
------------------------------
agent.py - implementation of question 2.

------------------------------
Q2 Commands
------------------------------
> /usr/local/lib/anaconda3-5.1.0/bin/python agent.py

------------------------------
Q2 Output Description
------------------------------
out/figures/agent_hits.png - the required histogram plot.

