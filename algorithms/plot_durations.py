import matplotlib.pyplot as plt

def plot_durations(original,trimmed):
    plt.hist(original, bins=50, alpha=0.7, label="Orginal Duration")
    plt.hist(trimmed, bins=50, alpha=0.7, label="Trimmed Duration")
    plt.xlabel("Value (seconds)")
    plt.ylabel("Frequency (count)")
    plt.title("Frequency Distribution of Audio Duration")
    plt.legend()
    plt.show()