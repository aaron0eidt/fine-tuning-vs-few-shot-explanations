import matplotlib.pyplot as plt
import numpy as np

# Define groups of metrics and scores
metrics_group1 = [
    "Kendall Tau\n(GT vs T5 Generated)",
    "Kendall Tau\n(GT vs LLaMA Generated)"
]
scores_group1 = [0.17367, 0.59793]

metrics_group2 = [
    "Kendall Tau\n(GT vs T5 Human)",
    "Kendall Tau\n(GT vs LLaMA Human)"
]
scores_group2 = [0.24651, 0.07748]

metric_krippendorf = ["Krippendorff's\nAlpha"]
score_krippendorf = [0.30253]

width = 0.6  # Bar width
fontsize_all = 16  # Set a uniform font size for all elements

# Create a single figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# First subplot
x1 = np.arange(len(metrics_group1))  # Label locations for group 1
bars1 = axes[0].bar(x1, scores_group1, width, color='lightblue', alpha=0.8)

axes[0].set_ylabel('Average Scores', fontsize=fontsize_all)
axes[0].set_title('T5 Generated vs LLaMA Generated', fontsize=fontsize_all)
axes[0].set_xticks(x1)
axes[0].set_xticklabels(metrics_group1, rotation=0, ha='center', fontsize=fontsize_all)
axes[0].tick_params(axis='y', labelsize=fontsize_all)
axes[0].set_ylim(0, 1)

# Annotate bar values for group 1
for bar in bars1:
    height = bar.get_height()
    axes[0].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=fontsize_all)

# Second subplot
x2 = np.arange(len(metrics_group2))  # Label locations for group 2
bars2 = axes[1].bar(x2, scores_group2, width, color='coral', alpha=0.8)

axes[1].set_ylabel('Average Scores', fontsize=fontsize_all)
axes[1].set_title('T5 Human vs LLaMA Human', fontsize=fontsize_all)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(metrics_group2, rotation=0, ha='center', fontsize=fontsize_all)
axes[1].tick_params(axis='y', labelsize=fontsize_all)
axes[1].set_ylim(0, 1)

# Annotate bar values for group 2
for bar in bars2:
    height = bar.get_height()
    axes[1].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=fontsize_all)

# Third subplot for Krippendorff's Alpha
x3 = np.arange(len(metric_krippendorf))  # Label location for Krippendorff
bars3 = axes[2].bar(x3, score_krippendorf, width, color='green', alpha=0.8)

axes[2].set_ylabel('Average Scores', fontsize=fontsize_all)
axes[2].set_title("Krippendorff's Alpha", fontsize=fontsize_all)
axes[2].set_xticks(x3)
axes[2].set_xticklabels(metric_krippendorf, rotation=0, ha='center', fontsize=fontsize_all)
axes[2].tick_params(axis='y', labelsize=fontsize_all)
axes[2].set_ylim(0, 1)

# Annotate bar values for Krippendorff's Alpha
for bar in bars3:
    height = bar.get_height()
    axes[2].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=fontsize_all)

# Adjust layout
plt.tight_layout()
plt.show()
