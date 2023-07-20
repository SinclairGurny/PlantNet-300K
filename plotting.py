import torch
import pickle
import matplotlib.pyplot as plt

def create_plot(data):
    # Get columns
    trn = data['acc_train']
    val = data['acc_val']
    # Graph
    x = range(1, len(trn)+1)
    plt.plot(x, trn, label='train', marker='o')
    plt.plot(x, val, label='val', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid()
    plt.show()
    
def plot(model_name="xp_test"):
    # Load 
    path = f'results/{model_name}/{model_name}.pkl'
    with open(path, 'rb') as f:
        results = pickle.load(f)
    # Visualize Data
    create_plot(results)