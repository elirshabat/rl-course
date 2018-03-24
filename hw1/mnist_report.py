import os.path
import json
import matplotlib.pyplot as plt
import numpy as np

# Configurations
output_dir = "out"
figures_dir = os.path.join("out", "figures")
data_files = ["mnist_output_data.json", "mnist_optimized_output_data.json", "mnist_deep_output_data.json"]

# Create figures directory if it does not exist
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Loss curves
avg_loss_fig = plt.figure()

# Reading and reporting data
for file in data_files:
    path = os.path.join(output_dir, file)
    with open(path, 'rt') as f:
        data = json.load(f)
    
    # Loss curve
    plt.figure()
    batch = range(len(data['loss']['batch']))
    loss = data['loss']['loss']
    plt.plot(1 + np.array(batch), np.array(loss), label=data['type'])
    plt.title("Loss curve {} - batch-size:{}".format(data['type'], data['batch_size']))
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "loss_curve_{}".format(data['type'])))
    
    # Average loss curve
    plt.figure(avg_loss_fig.number)
    num_epoches = data['num_epochs']
    avg_loss = np.zeros(num_epoches)
    for epoch in range(num_epoches):
        num_batches = data['loss']['epoch'].count(epoch)
        loss  = [ data['loss']['loss'][i] for i in range(len(data['loss']['loss'])) if data['loss']['epoch'][i] == epoch ]
        avg_loss[epoch] = sum(loss)/len(loss)
    plt.plot(1 + np.array(range(num_epoches)), avg_loss, label=data['type'])
    plt.title("Average loss curve - batch-size:{}".format(data['batch_size']))
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.legend()
    
    # Accuracy
    print('Accuracy of {} mnist on the 10000 test images: {}'.format(data['type'], data['accuracy']))

# Save average loss curve figure
plt.figure(avg_loss_fig.number)
plt.savefig(os.path.join(figures_dir, "average_loss_curve"))

plt.show()
