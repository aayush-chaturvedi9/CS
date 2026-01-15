import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
path = "supply_chain.csv"   # <-- change to your file name if needed
df = pd.read_csv(path)

# Convert dates
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"])

# Compute Total Lead Time
df["Total_Lead_Time"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days

# --- 1. Lead Time per Order ---
plt.figure(figsize=(7,4))
plt.plot(df["Order_ID"], df["Total_Lead_Time"])
plt.xlabel("Order ID")
plt.ylabel("Total Lead Time (Days)")
plt.title("Lead Time per Order")
plt.grid(True)
plt.tight_layout()
plt.savefig("lead_time_per_order.png")
plt.show()

# --- 2. Cost vs Lead Time ---
plt.figure(figsize=(7,4))
plt.scatter(df["Delivery_Cost"], df["Total_Lead_Time"])
plt.xlabel("Delivery Cost")
plt.ylabel("Total Lead Time (Days)")
plt.title("Cost vs Lead Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_vs_lead_time.png")
plt.show()

# --- 3. Breakdown of Process Times by Stage ---
plt.figure(figsize=(7,4))
plt.plot(df["Order_ID"], df["Warehouse_Time"], label="Warehouse")
plt.plot(df["Order_ID"], df["Transport_Time"], label="Transport")
plt.plot(df["Order_ID"], df["Last_Mile_Time"], label="Last Mile")
plt.xlabel("Order ID")
plt.ylabel("Time (Days)")
plt.title("Process Time Breakdown by Stage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("process_time_breakdown.png")
plt.show()

# --- 4. Bottleneck Variance Chart ---
stage_variances = {
    "Warehouse": np.var(df["Warehouse_Time"]),
    "Transport": np.var(df["Transport_Time"]),
    "Last Mile": np.var(df["Last_Mile_Time"])
}

plt.figure(figsize=(6,4))
plt.bar(stage_variances.keys(), stage_variances.values())
plt.xlabel("Stage")
plt.ylabel("Variance (Days^2)")
plt.title("Bottleneck Detection by Variance")
plt.tight_layout()
plt.savefig("bottleneck_variance.png")
plt.show()

# --- 5. Inventory vs Demand ---
plt.figure(figsize=(7,4))
plt.plot(df["Order_ID"], df["Demand"], label="Demand")
plt.plot(df["Order_ID"], df["Inventory_Level"], label="Inventory Level")
plt.xlabel("Order ID")
plt.ylabel("Units")
plt.title("Demand vs Inventory Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("inventory_vs_demand.png")
plt.show()

# --- 6. Efficiency Score Calculation ---
mean_lead = df["Total_Lead_Time"].mean()
df["Lead_Time_Deviation"] = df["Total_Lead_Time"] - mean_lead

df["Efficiency_Score"] = 100 - (
    (abs(df["Lead_Time_Deviation"]) / abs(df["Lead_Time_Deviation"]).max()) * 50 +
    (df["Delivery_Cost"] / df["Delivery_Cost"].max()) * 50
)

df["Efficiency_Score"] = df["Efficiency_Score"].clip(0, 100)

plt.figure(figsize=(8,4))
plt.bar(df["Order_ID"][:20], df["Efficiency_Score"][:20])
plt.xlabel("Order ID")
plt.ylabel("Efficiency Score")
plt.title("Top 20 Efficiency Scores")
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_scores.png")
plt.show()

# --- 7. Risk Classification ---
df["Risk"] = np.where(
    (df["Lead_Time_Deviation"] > 2) & (df["Inventory_Level"] < df["Demand"]),
    "High Risk",
    "Low Risk"
)

risk_counts = df["Risk"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(risk_counts.index, risk_counts.values)
plt.xlabel("Risk Level")
plt.ylabel("Number of Orders")
plt.title("High Risk vs Low Risk Orders")
plt.tight_layout()
plt.savefig("risk_levels.png")
plt.show()

print("All graphs generated and saved successfully!")
