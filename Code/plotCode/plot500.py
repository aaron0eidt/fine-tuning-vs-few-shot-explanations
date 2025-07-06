import matplotlib.pyplot as plt
import numpy as np

# Metrics and scores
metrics = ['Similarity', 'BERTScore', 'BLEURT', 'Kendall Tau']
t5_scores = [0.64706, 0.8644, -0.95572, 0.17058]
llama_scores = [0.58832, 0.84866, -0.52462, 0.54281]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, t5_scores, width, label='T5', alpha=0.8)
bars2 = ax.bar(x + width/2, llama_scores, width, label='LLaMA', alpha=0.8)

ax.set_ylabel('Scores')
ax.set_title('Performance Comparison: T5 vs LLaMA')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(-1.0, 1.0)  # Extend y-axis to include 1.0
ax.legend()

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('plot500.png')  # Save the updated figure
plt.show()
