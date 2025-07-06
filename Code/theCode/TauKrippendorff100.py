import pandas as pd
from scipy.stats import kendalltau
import krippendorff
import ast

# Load the CSV file
input_csv = "output22222.csv"  # Replace with your cleaned file path
output_csv = "100_FINAL.csv"

df = pd.read_csv(input_csv)

# Define the ratings columns
ratings_columns = ["Ground Truth Ratings", "T5 Ratings", "LLaMA Ratings", "T5 Human Ratings", "LLaMA Human Ratings"]

# Extract ratings from JSON-like strings
def extract_ratings(json_string):
    try:
        json_dict = ast.literal_eval(json_string)
        return list(json_dict.values())  # Extract all values as a list
    except (ValueError, SyntaxError):
        return []

# Compute Kendall's Tau row-wise between two columns
def rowwise_kendall(row, col1, col2):
    try:
        ratings1 = extract_ratings(row[col1])
        ratings2 = extract_ratings(row[col2])
        tau, _ = kendalltau(ratings1, ratings2)
        return tau
    except Exception:
        return None

# Add row-wise Kendall's Tau columns
df["Kendall Tau (Ground Truth vs T5)"] = df.apply(
    lambda row: rowwise_kendall(row, "Ground Truth Ratings", "T5 Ratings"), axis=1
)
df["Kendall Tau (Ground Truth vs LLaMA)"] = df.apply(
    lambda row: rowwise_kendall(row, "Ground Truth Ratings", "LLaMA Ratings"), axis=1
)
df["Kendall Tau (Ground Truth vs T5 Human)"] = df.apply(
    lambda row: rowwise_kendall(row, "Ground Truth Ratings", "T5 Human Ratings"), axis=1
)
df["Kendall Tau (Ground Truth vs LLaMA Human)"] = df.apply(
    lambda row: rowwise_kendall(row, "Ground Truth Ratings", "LLaMA Human Ratings"), axis=1
)

# Prepare data for Krippendorff's Alpha
def prepare_data_for_krippendorff(df, columns):
    ratings_matrix = []
    for col in columns:
        col_ratings = df[col].apply(extract_ratings).tolist()
        for row_idx, rating_list in enumerate(col_ratings):
            if row_idx >= len(ratings_matrix):
                ratings_matrix.append([])
            ratings_matrix[row_idx].append(rating_list if rating_list else [None])
    return ratings_matrix

# Calculate Krippendorff's Alpha
def calculate_krippendorff_alpha(matrix):
    try:
        flattened_matrix = []
        for row in matrix:
            flattened_row = []
            for cell in row:
                flattened_row.append(cell[0] if cell else None)
            flattened_matrix.append(flattened_row)

        # Filter out rows with all None
        valid_matrix = [row for row in flattened_matrix if any(x is not None for x in row)]
        return krippendorff.alpha(reliability_data=valid_matrix, level_of_measurement="ordinal")
    except Exception as e:
        print(f"Error calculating Krippendorff's Alpha: {e}")
        return None

# Calculate Krippendorff's Alpha
kripp_alpha = calculate_krippendorff_alpha(prepare_data_for_krippendorff(df, ratings_columns))
print(f"Krippendorff's Alpha: {kripp_alpha}")

# Calculate averages for the new Kendall's Tau columns
tau_t5_avg = df["Kendall Tau (Ground Truth vs T5)"].mean()
tau_llama_avg = df["Kendall Tau (Ground Truth vs LLaMA)"].mean()
tau_t5_human_avg = df["Kendall Tau (Ground Truth vs T5 Human)"].mean()
tau_llama_human_avg = df["Kendall Tau (Ground Truth vs LLaMA Human)"].mean()

# Find the "Average Scores" row and update it with the averages
average_row_index = df[df.iloc[:, 0] == "Average Scores"].index
if not average_row_index.empty:
    average_row_index = average_row_index[0]
    df.at[average_row_index, "Kendall Tau (Ground Truth vs T5)"] = tau_t5_avg
    df.at[average_row_index, "Kendall Tau (Ground Truth vs LLaMA)"] = tau_llama_avg
    df.at[average_row_index, "Kendall Tau (Ground Truth vs T5 Human)"] = tau_t5_human_avg
    df.at[average_row_index, "Kendall Tau (Ground Truth vs LLaMA Human)"] = tau_llama_human_avg
    df.at[average_row_index, "Krippendorff's Alpha"] = kripp_alpha
else:
    # Insert the row with the correct number of columns
    num_columns = df.shape[1]
    average_scores_row = ["Average Scores"] + [None] * (num_columns - 6) + [tau_t5_avg, tau_llama_avg, tau_t5_human_avg, tau_llama_human_avg, kripp_alpha]
    df.loc[len(df)] = average_scores_row

# Save the updated DataFrame
df.to_csv(output_csv, index=False)

print(f"Updated CSV with row-wise Kendall's Tau, Krippendorff's Alpha, and averages saved as {output_csv}")
