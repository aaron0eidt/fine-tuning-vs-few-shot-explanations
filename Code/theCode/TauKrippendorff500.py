import pandas as pd
from scipy.stats import kendalltau
import krippendorff
import ast

# Load the CSV file
input_csv = "output.csv"  # Replace with your cleaned file path
output_csv = "output_with_row_tests_and_averages.csv"

df = pd.read_csv(input_csv)

# Define the ratings columns
ratings_columns = ["Ground Truth Ratings", "T5 Ratings", "LLaMA Ratings"]

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

# Prepare data for Krippendorff's Alpha
def prepare_data_for_krippendorff(df, columns):
    ratings_matrix = []
    for col in columns:
        col_ratings = df[col].apply(extract_ratings).tolist()
        for row_idx, rating_list in enumerate(col_ratings):
            if row_idx >= len(ratings_matrix):
                ratings_matrix.append([])
            # Extend the row with the ratings list (pad with None for missing values)
            if rating_list:
                ratings_matrix[row_idx].append(rating_list)
            else:
                ratings_matrix[row_idx].append(None)
    return ratings_matrix

# Debugging to check the structure of the ratings matrix
ratings_matrix = prepare_data_for_krippendorff(df, ratings_columns)
print("Ratings Matrix (for Krippendorff's Alpha):")
for row in ratings_matrix[:5]:  # Print the first 5 rows for inspection
    print(row)

# Calculate Krippendorff's Alpha
def calculate_krippendorff_alpha(matrix):
    try:
        # Convert matrix to Krippendorff-compatible format
        flattened_matrix = []
        for row in matrix:
            flattened_row = []
            for cell in row:
                if cell is not None and isinstance(cell, list):
                    flattened_row.extend(cell)  # Flatten nested lists
                elif cell is not None:
                    flattened_row.append(cell)
            flattened_matrix.append(flattened_row)

        # Filter out rows with all None
        valid_matrix = [row for row in flattened_matrix if any(x is not None for x in row)]
        return krippendorff.alpha(reliability_data=valid_matrix, level_of_measurement="ordinal")
    except Exception as e:
        print(f"Error calculating Krippendorff's Alpha: {e}")
        return None

kripp_alpha = calculate_krippendorff_alpha(ratings_matrix)
print(f"Krippendorff's Alpha: {kripp_alpha}")

# Calculate averages for the new Kendall's Tau columns
tau_t5_avg = df["Kendall Tau (Ground Truth vs T5)"].mean()
tau_llama_avg = df["Kendall Tau (Ground Truth vs LLaMA)"].mean()

# Find the "Average Scores" row and update it with the averages
average_row_index = df[df.iloc[:, 0] == "Average Scores"].index
if not average_row_index.empty:
    average_row_index = average_row_index[0]
    df.at[average_row_index, "Kendall Tau (Ground Truth vs T5)"] = tau_t5_avg
    df.at[average_row_index, "Kendall Tau (Ground Truth vs LLaMA)"] = tau_llama_avg
    df.at[average_row_index, "Krippendorff's Alpha"] = kripp_alpha
else:
    # If "Average Scores" row does not exist, append it as a new row
    df.loc[len(df)] = ["Average Scores"] + [None] * (df.shape[1] - 4) + [tau_t5_avg, tau_llama_avg, kripp_alpha]

# Save the updated DataFrame
df.to_csv(output_csv, index=False)

print(f"Updated CSV with row-wise Kendall's Tau, Krippendorff's Alpha, and averages saved as {output_csv}")
