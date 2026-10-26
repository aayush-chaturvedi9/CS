
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Restaurant Order Pattern Analysis

DATA_FILE = "restaurant_orders_dataset.csv"
RESULTS_FOLDER = "Screenshots_Results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 1. Load dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "CSV file not found. Keep both files in the same folder."
    )

df = pd.read_csv(DATA_FILE)

# 2. Check required columns
required_columns = [
    "Order_ID", "Customer_ID", "Food_Item", "Category",
    "Quantity", "Price", "Order_Date", "Order_Hour",
    "Payment_Method"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 3. Data preprocessing
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

for col in ["Quantity", "Price", "Order_Hour"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=required_columns, inplace=True)

df["Revenue"] = df["Quantity"] * df["Price"]
df["Day"] = df["Order_Date"].dt.day_name()
df["Month"] = df["Order_Date"].dt.month_name()

df["Time_Period"] = pd.cut(
    df["Order_Hour"],
    bins=[9, 12, 17, 21, 23],
    labels=["Morning", "Afternoon", "Evening", "Night"],
    include_lowest=True
)

if df.empty:
    raise ValueError("No valid records found in the dataset.")

print("\nRESTAURANT ORDER PATTERN ANALYSIS")
print("=" * 45)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df[["Quantity", "Price", "Order_Hour", "Revenue"]].describe())

# 4. Food item analysis
popular_items = (
    df.groupby("Food_Item")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

food_revenue = (
    df.groupby("Food_Item")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nMOST POPULAR FOOD ITEMS")
print(popular_items)

print("\nREVENUE BY FOOD ITEM")
print(food_revenue)

# 5. Category analysis
category_orders = (
    df.groupby("Category")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

category_revenue = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nCATEGORY-WISE QUANTITY")
print(category_orders)

print("\nCATEGORY-WISE REVENUE")
print(category_revenue)

# 6. Customer analysis
customer_orders = df.groupby("Customer_ID")["Order_ID"].count()

customer_spending = (
    df.groupby("Customer_ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nCUSTOMER ORDER COUNTS")
print(customer_orders)

print("\nCUSTOMER SPENDING")
print(customer_spending)

# 7. Payment analysis
payment_analysis = df["Payment_Method"].value_counts()

print("\nPAYMENT METHOD ANALYSIS")
print(payment_analysis)

# 8. Day-wise analysis
days = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

day_orders = (
    df.groupby("Day")["Quantity"]
    .sum()
    .reindex(days)
    .fillna(0)
)

print("\nDAY-WISE QUANTITY SOLD")
print(day_orders)

# 9. Time and hourly analysis
hourly_orders = (
    df.groupby("Order_Hour")["Order_ID"]
    .count()
    .sort_index()
)

time_analysis = df.groupby(
    "Time_Period", observed=True
)["Order_ID"].count()

print("\nORDERS BY TIME PERIOD")
print(time_analysis)

print("\nHOURLY ORDER COUNTS")
print(hourly_orders)

# 10. Final results
best_item = popular_items.idxmax()
highest_revenue_item = food_revenue.idxmax()
peak_hour = int(hourly_orders.idxmax())

total_revenue = df["Revenue"].sum()
average_order_value = df["Revenue"].mean()

top_customer = customer_spending.idxmax()
top_customer_spending = customer_spending.max()

print("\nFINAL ANALYSIS")
print("=" * 45)

print("Best Selling Food:", best_item)
print("Highest Revenue Item:", highest_revenue_item)
print("Peak Ordering Hour:", str(peak_hour) + ":00")
print("Orders During Peak Hour:", hourly_orders.max())
print("Total Revenue: Rs.", round(total_revenue, 2))
print("Average Revenue Per Record: Rs.", round(average_order_value, 2))
print("Top Customer:", top_customer)
print("Top Customer Spending: Rs.", round(top_customer_spending, 2))
print("Most Popular Category:", category_orders.idxmax())
print("Most Used Payment Method:", payment_analysis.idxmax())
print("Most Active Day:", day_orders.idxmax())

# 11. Graph 1 - Popular food items
sns.set_theme(style="whitegrid")

popular_items.head(10).plot(kind="bar", figsize=(10, 6))
plt.title("Top 10 Most Ordered Food Items")
plt.xlabel("Food Item")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "01_top_food_items.png"))
plt.show()
plt.close()

# 12. Graph 2 - Category orders
category_orders.plot(kind="bar", figsize=(9, 5))
plt.title("Quantity Sold by Category")
plt.xlabel("Category")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "02_category_orders.png"))
plt.show()
plt.close()

# 13. Graph 3 - Category revenue
category_revenue.plot(kind="bar", figsize=(9, 5))
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "03_category_revenue.png"))
plt.show()
plt.close()

# 14. Graph 4 - Payment methods
payment_analysis.plot(
    kind="pie", figsize=(7, 7), autopct="%1.1f%%"
)
plt.title("Payment Method Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "04_payment_methods.png"))
plt.show()
plt.close()

# 15. Graph 5 - Day-wise orders
day_orders.plot(kind="bar", figsize=(10, 5))
plt.title("Quantity Sold by Day of Week")
plt.xlabel("Day")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "05_day_orders.png"))
plt.show()
plt.close()

# 16. Graph 6 - Hourly order pattern
hourly_orders.plot(
    kind="line", figsize=(10, 5), marker="o"
)
plt.title("Restaurant Orders by Hour")
plt.xlabel("Hour")
plt.ylabel("Number of Orders")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "06_hourly_orders.png"))
plt.show()
plt.close()

# 17. Graph 7 - Time period
time_analysis.plot(kind="bar", figsize=(8, 5))
plt.title("Orders by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "07_time_period.png"))
plt.show()
plt.close()

# 18. Graph 8 - Correlation heatmap
numeric_data = df[
    ["Quantity", "Price", "Order_Hour", "Revenue"]
]

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Restaurant Order Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "08_correlation_heatmap.png"))
plt.show()
plt.close()

# 19. Save results in a text file
with open(
    os.path.join(RESULTS_FOLDER, "analysis_results.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write("RESTAURANT ORDER PATTERN ANALYSIS\n\n")
    file.write(f"Records Analysed: {len(df)}\n")
    file.write(f"Best Selling Food: {best_item}\n")
    file.write(f"Highest Revenue Item: {highest_revenue_item}\n")
    file.write(f"Peak Ordering Hour: {peak_hour}:00\n")
    file.write(f"Total Revenue: Rs. {total_revenue:.2f}\n")
    file.write(
        f"Average Revenue Per Record: Rs. {average_order_value:.2f}\n"
    )
    file.write(f"Top Customer: {top_customer}\n")
    file.write(
        f"Top Customer Spending: Rs. {top_customer_spending:.2f}\n"
    )
    file.write(f"Most Popular Category: {category_orders.idxmax()}\n")
    file.write(
        f"Most Used Payment Method: {payment_analysis.idxmax()}\n"
    )
    file.write(f"Most Active Day: {day_orders.idxmax()}\n")

print("\nGraphs and results saved in:", RESULTS_FOLDER)
print("Analysis completed successfully!")
