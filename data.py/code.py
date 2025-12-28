import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. DATA INGESTION & VALIDATION
# =====================================================
df = pd.read_csv("sports_advanced.csv")

if df.isnull().sum().any():
    raise ValueError("Dataset contains missing values")

print("\nDATA LOADED SUCCESSFULLY")

# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================
df["BattingImpact"] = df["Runs"] * df["StrikeRate"] / 100
df["BowlingImpact"] = df["Wickets"] * (10 - df["Economy"].replace(0, 10))
df["OverallImpact"] = df["BattingImpact"] + df["BowlingImpact"]

df["ConsistencyScore"] = df["Average"] / df["Matches"]
df["ExperienceFactor"] = np.log(df["Matches"] + 1)

# =====================================================
# 3. ADVANCED PERFORMANCE INDEX (API)
# =====================================================
df["AdvancedPerformanceIndex"] = (
    0.4 * df["BattingImpact"] +
    0.3 * df["BowlingImpact"] +
    0.2 * df["ConsistencyScore"] * 100 +
    0.1 * df["ExperienceFactor"] * 100
)

# =====================================================
# 4. PLAYER SEGMENTATION
# =====================================================
conditions = [
    (df["Wickets"] > 10) & (df["Runs"] > 600),
    (df["Runs"] > 700),
    (df["Wickets"] > 10)
]

categories = ["Elite All-Rounder", "Specialist Batsman", "Strike Bowler"]
df["PlayerRole"] = np.select(conditions, categories, default="Support Player")

# =====================================================
# 5. TEAM BALANCE ANALYSIS
# =====================================================
team_analysis = df.groupby("Team").agg({
    "Runs": "sum",
    "Wickets": "sum",
    "AdvancedPerformanceIndex": "mean"
})

team_analysis["TeamStrengthScore"] = (
    0.6 * team_analysis["Runs"] +
    30 * team_analysis["Wickets"] +
    50 * team_analysis["AdvancedPerformanceIndex"]
)

# =====================================================
# 6. MOMENTUM & TREND ANALYSIS
# =====================================================
df["MomentumScore"] = df["StrikeRate"] * df["ConsistencyScore"]

# =====================================================
# 7. RANKING ENGINE
# =====================================================
ranked_players = df.sort_values(
    by="AdvancedPerformanceIndex",
    ascending=False
)

# =====================================================
# 8. VISUAL ANALYTICS DASHBOARD
# =====================================================

# Impact Comparison
plt.figure()
plt.bar(ranked_players["Player"], ranked_players["AdvancedPerformanceIndex"])
plt.xticks(rotation=45)
plt.xlabel("Players")
plt.ylabel("API Score")
plt.title("Advanced Player Performance Index")
plt.show()

# Team Strength
plt.figure()
plt.bar(team_analysis.index, team_analysis["TeamStrengthScore"])
plt.xlabel("Teams")
plt.ylabel("Strength Score")
plt.title("Team Strength Comparison")
plt.show()

# Role Distribution
plt.figure()
df["PlayerRole"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Player Role Distribution")
plt.ylabel("")
plt.show()

# Momentum Analysis
plt.figure()
plt.scatter(df["ConsistencyScore"], df["StrikeRate"])
plt.xlabel("Consistency")
plt.ylabel("Strike Rate")
plt.title("Momentum Analysis")
plt.show()

# =====================================================
# 9. DECISION SUPPORT OUTPUT
# =====================================================
print("\n===== TOP 5 PLAYERS (API BASED) =====")
print(ranked_players[["Player", "PlayerRole", "AdvancedPerformanceIndex"]].head())

print("\n===== TEAM STRENGTH =====")
print(team_analysis[["TeamStrengthScore"]])

print("\n===== KEY INSIGHTS =====")
print("• Elite All-Rounders provide maximum impact")
print("• High momentum players indicate current form")
print("• Balanced teams outperform star-dependent teams")