import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the generated evaluation file
df = pd.read_csv("test_reports/documents.csv")

# Set up a clean visualization canvas
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Reshape data for plotting predicted vs expected rows side-by-side
df_melted = df.melt(
    id_vars=["document_id"],
    value_vars=["predicted_row_count", "expected_row_count"],
    var_name="Row Type",
    value_name="Count"
)

# Create a bar plot comparing the row outputs
ax = sns.barplot(x="Count", y="document_id", hue="Row Type", data=df_melted, palette="muted")

plt.title("OCR Model Output vs. Human Ground Truth Row Counts", fontsize=14, fontweight="bold")
plt.xlabel("Number of Table Rows", fontsize=12)
plt.ylabel("Document ID", fontsize=12)
plt.tight_layout()

# Save the plot image
plt.savefig("test_reports/model_performance_chart.png", dpi=300)
print("[+] Performance comparison chart generated at: test_reports/model_performance_chart.png")