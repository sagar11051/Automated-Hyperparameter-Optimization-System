import matplotlib.pyplot as plt

def plot_learning_curves(history, title):
    iterations = range(len(history))
    scores = [score for params, score in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, scores, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('ROC AUC Score')
    plt.grid(True)
    plt.show()
