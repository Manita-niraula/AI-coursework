import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def divider():
    print("-" * 60)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────
section("1. LOAD DATASET")

df = pd.read_csv(r"C:\Users\LPR HUB\Documents\Third sem\AI\Ai final project\all countries global temperature.csv")

print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"  Years   : {df.columns[4]} – {df.columns[-1]}")

print("\n  Preview of RAW data (first 5 rows, selected columns):")
divider()
preview_cols = ["ObjectId", "Country Name", "Unit", "Change ", df.columns[4], df.columns[5], df.columns[-2], df.columns[-1]]
print(df[preview_cols].head())
divider()


# ─────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────
section("2. DATA CLEANING")

# Year columns (dynamic)
year_cols = list(df.columns[4:])   # e.g., '1970' ... '2021'

# Missing values BEFORE cleaning
missing_before = df[year_cols].isnull().sum().sum()
print(f"\n  Missing values BEFORE cleaning (year columns only): {missing_before}")

print("\n  Missing per-year (top 10 years with most missing):")
divider()
missing_by_year = df[year_cols].isnull().sum().sort_values(ascending=False)
print(missing_by_year.head(10))
divider()

# Convert to numeric (ensure interpolation works)
df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

# Interpolate across years (row-wise)
df[year_cols] = df[year_cols].interpolate(method="linear", axis=1, limit_direction="both")

# Missing values AFTER cleaning
missing_after = df[year_cols].isnull().sum().sum()
print(f"\n  Missing values AFTER cleaning (year columns only):  {missing_after}")
print("  ✔ Interpolation complete")

print("\n  Preview of CLEANED data (first 5 rows, selected columns):")
divider()
print(df[preview_cols].head())
divider()

# ─────────────────────────────────────────────────────────────
# 3. EXPLORATORY ANALYSIS
# ─────────────────────────────────────────────────────────────
section("3. EXPLORATORY ANALYSIS")
year_cols = df.columns[4:]

max_val     = df[year_cols].max().max()
max_year    = df[year_cols].max().idxmax()
max_country = df.loc[df[max_year].idxmax(), "Country Name"]

min_val     = df[year_cols].min().min()
min_year    = df[year_cols].min().idxmin()
min_country = df.loc[df[min_year].idxmin(), "Country Name"]

print("\n  Single-Year Extremes:")
divider()
print(f"  {'Metric':<35} {'Value':>10}  {'Year':>6}  Country")
divider()
print(f"  {'Highest Temperature Rise':<35} {max_val:>10.4f}  {max_year:>6}  {max_country}")
print(f"  {'Lowest Temperature Drop':<35} {min_val:>10.4f}  {min_year:>6}  {min_country}")
divider()

yearly_mean  = df[year_cols].mean()
global_avg   = yearly_mean.mean()
country_mean = df.set_index("Country Name")[year_cols].mean(axis=1)

# Cache sorted versions to avoid repeated sorting
country_mean_sorted_desc = country_mean.sort_values(ascending=False)
country_mean_sorted_asc  = country_mean.sort_values(ascending=True)

print("\n  Country Averages (1970–2021):")
divider()
print(f"  {'Overall Global Average':<40} {global_avg:>8.4f} °C")
print(f"  {'Highest Average Rise':<40} {country_mean.max():>8.4f} °C  →  {country_mean.idxmax()}")
print(f"  {'Lowest Average Drop':<40} {country_mean.min():>8.4f} °C  →  {country_mean.idxmin()}")
divider()

print("\n  Global Yearly Mean Temperature Change:")
divider()
print(f"  {'Year':<8} {'Avg Temp Change (°C)':>22}")
divider()
for yr, val in yearly_mean.items():
    print(f"  {yr:<8} {val:>22.4f}")
divider()

temp_change      = df[year_cols].subtract(df["1970"], axis=0)
df["max_rise"]   = temp_change.max(axis=1)
countries_over_2 = df[df["max_rise"] > 2]["Country Name"].tolist()

print(f"\n  Countries with >2°C rise above 1970 baseline:")
divider()
print(f"  Total : {len(countries_over_2)}")
for i, c in enumerate(countries_over_2, 1):
    print(f"  {i:>3}. {c}")
divider()


# ─────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
section("4. VISUALISATIONS  (charts will open one by one)")

# --- Line chart ---
plt.figure(figsize=(14, 6))
plt.plot(yearly_mean.index, yearly_mean.values, marker="o", color="steelblue")
plt.title("Global Yearly Mean Temperature Change (1970–2021)")
plt.xlabel("Year"); plt.ylabel("Temperature Change (°C)")
plt.xticks(rotation=90); plt.grid(True); plt.tight_layout(); plt.show()

# --- Bar chart ---
plt.figure(figsize=(14, 6))
plt.bar(yearly_mean.index, yearly_mean.values, color="skyblue")
plt.title("Global Yearly Mean Temperature Change (1970–2021)")
plt.xlabel("Year"); plt.ylabel("Temperature Change (°C)")
plt.xticks(rotation=90); plt.tight_layout(); plt.show()

# --- Scatter: country averages ---
plt.figure(figsize=(14, 6))
plt.scatter(country_mean.index, country_mean.values, alpha=0.7, color="darkorange")
plt.title("Country-wise Mean Temperature Change (1970–2021)")
plt.xlabel("Country"); plt.ylabel("Mean Temp Change (°C)")
plt.xticks([], []); plt.tight_layout(); plt.show()

# --- All-country spaghetti lines ---
plt.figure(figsize=(14, 42))
for _, row in df.iterrows():
    plt.plot(year_cols, row[year_cols].values, alpha=0.3, linewidth=0.6)
plt.title("Temperature Change Trends of All Countries (1970–2021)")
plt.xlabel("Year"); plt.ylabel("Temperature Change (°C)")
plt.xticks(rotation=90);plt.subplots_adjust(top=0.96); plt.show()


# --- All-country Heat Map --- 
n = len(country_mean_sorted_desc)

row_height = 0.55
fig_height = max(15, n * row_height)

fig, ax = plt.subplots(figsize=(14, fig_height))

values = country_mean_sorted_desc.values.reshape(-1, 1)

im = ax.imshow(values, cmap="RdYlBu_r", aspect="auto",
               vmin=values.min(), vmax=values.max())

step = 5   
ax.set_yticks(range(0, n, step))
ax.set_yticklabels(country_mean_sorted_desc.index[::step], fontsize=9)

ax.set_xticks([0])
ax.set_xticklabels(["Mean ΔT (°C)"], fontsize=11)

for spine in ax.spines.values():
    spine.set_visible(False)

cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cbar.set_label("Mean Temperature Change (°C)", fontsize=11)

ax.set_title("Heatmap of Countries by Mean Temperature Change (1970–2021)",
             fontsize=14, pad=15)

plt.subplots_adjust(left=0.35, right=0.92, top=0.95, bottom=0.05)
plt.show()


# --- Top 10 / Bottom 10 bar chart ---
top10    = country_mean_sorted_desc.head(10)
bottom10 = country_mean_sorted_asc.head(10)

plt.figure(figsize=(14, 6))
plt.bar(top10.index, top10.values, color="red", label="Most Affected")
plt.bar(bottom10.index, bottom10.values, color="blue", label="Least Affected")
plt.title("Top 10 and Bottom 10 Countries by Mean Temperature Change")
plt.xlabel("Country"); plt.ylabel("Mean Temp Change (°C)")
plt.xticks(rotation=90); plt.legend(); plt.tight_layout(); plt.show()

# --- Boxplot by year ---
plt.figure(figsize=(14, 6))
plt.boxplot([df[year].dropna().values for year in year_cols])
plt.ylabel("Temperature Change (°C)")
plt.title("Yearly Distribution of Temperature Changes Across Countries")
plt.xticks(range(1, len(year_cols) + 1, 5), year_cols[::5], rotation=90)
plt.tight_layout(); plt.show()

# --- Pie chart (only positive values to avoid errors) ---
top10_pie = country_mean_sorted_desc.head(10)
top10_pie = top10_pie[top10_pie > 0]   # guard against negatives
plt.figure(figsize=(8, 8))
plt.pie(top10_pie, labels=top10_pie.index, autopct='%1.1f%%')
plt.title("Top 10 Most Affected Countries"); plt.show()


# ─────────────────────────────────────────────────────────────
# 5. GLOBAL LINEAR REGRESSION MODEL 
# ─────────────────────────────────────────────────────────────
section("5. GLOBAL LINEAR REGRESSION MODEL (TRAIN/TEST)")

# Build global yearly mean dataset
global_data = pd.DataFrame({
    "Year": yearly_mean.index.astype(int),
    "Temperature": yearly_mean.values
}).sort_values("Year").reset_index(drop=True)

# Time-based split (better for time series)
train_data = global_data[global_data["Year"] <= 2010]
test_data  = global_data[global_data["Year"] > 2010]

X_train = train_data[["Year"]]
y_train = train_data["Temperature"]

X_test = test_data[["Year"]]
y_test = test_data["Temperature"]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

plt.scatter(X_train["Year"], y_train, label="Training Data (1970–2010)")
plt.plot(X_train["Year"],
         model.predict(X_train),
         color="red", label="Fitted Regression Line")
plt.title("Linear Regression Model Training (1970–2010)")
plt.xlabel("Year")
plt.ylabel("Temperature Change (°C)")
plt.legend()
plt.show()

# Predict on test set
predictions = model.predict(X_test)

# Metrics
from sklearn.metrics import mean_absolute_error
mse  = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

print("\n  Model Performance (Train: 1970–2010, Test: 2011–2021):")
divider()
print(f"  {'Intercept (β₀)':<30} {model.intercept_:>12.6f}")
print(f"  {'Slope (β₁)':<30} {model.coef_[0]:>12.6f}")
print(f"  {'Mean Squared Error':<30} {mse:>12.6f}")
print(f"  {'Root Mean Squared Error':<30} {rmse:>12.6f}")
print(f"  {'Mean Absolute Error':<30} {mae:>12.6f}")
print(f"  {'R² Score':<30} {r2:>12.4f}")
divider()

print("\n  Actual vs Predicted (Test Set):")
divider()
print(f"  {'Year':<8} {'Actual (°C)':>14} {'Predicted (°C)':>16} {'Error':>10}")
divider()
for yr, actual, pred in zip(X_test["Year"].values, y_test.values, predictions):
    print(f"  {yr:<8} {actual:>14.4f} {pred:>16.4f} {actual - pred:>10.4f}")
divider()
# Actual vs Predicted (Test Data Only)

plt.figure(figsize=(10,6))

plt.plot(X_test["Year"], y_test, marker='o', label="Actual Temperature (2011–2021)")
plt.plot(X_test["Year"], predictions, marker='o', linestyle='--',
         label="Predicted Temperature (2011–2021)")

plt.title("Actual vs Predicted Global Temperature Change (Test Set)")
plt.xlabel("Year")
plt.ylabel("Temperature Change (°C)")
plt.legend()
plt.grid(True)

plt.show()

# Plot: train, test, and regression line (full range)
plt.figure(figsize=(12, 6))

plt.scatter(X_train["Year"], y_train, label="Train Data (1970–2010)", alpha=0.7)
plt.scatter(X_test["Year"], y_test, label="Test Data (2011–2021)", alpha=0.7)

# Regression line from start to end of available data
X_line = np.arange(global_data["Year"].min(), global_data["Year"].max() + 1).reshape(-1, 1)
plt.plot(X_line.flatten(),model.predict(pd.DataFrame(X_line, columns=["Year"])),linewidth=2, label="Regression Line")
plt.title("Simple Linear Regression: Global Temperature Change (Train/Test Split)")
plt.xlabel("Year")
plt.ylabel("Temperature Change (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

 #─────────────────────────────────────────────────────────────
# ADDITIONAL MODEL EVALUATION VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
# section("5A. RESIDUAL & ERROR ANALYSIS")

# Calculate residuals
residuals = y_test - predictions

# --- Residual Plot ---
plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals)
plt.axhline(0)
plt.title("Residual Plot (Test Data)")
plt.xlabel("Predicted Temperature (°C)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Error Distribution Plot ---
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=8)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# Re-train on FULL data (1970–2021) for forecasting after evaluation
model.fit(global_data[["Year"]], global_data["Temperature"])
print("\n Model re-trained on full data (1970–2021) for forecasting")


# ─────────────────────────────────────────────────────────────
# 6. PER-COUNTRY FUTURE PREDICTIONS (2022–2050)
# ─────────────────────────────────────────────────────────────
section("6. PER-COUNTRY PREDICTIONS (2022–2050)")
print("  Training individual models for each country...")

future_years = list(range(2022, 2051))
results = []

X_years = pd.DataFrame([int(yr) for yr in year_cols], columns=["Year"])
n_years = len(year_cols)
split_idx = int(n_years * 0.8)

for _, row in df.iterrows():
    country = row["Country Name"]
    y_vals  = row[year_cols].values

    X_tr = X_years.iloc[:split_idx]
    y_tr = y_vals[:split_idx]

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    future_preds = lr.predict(pd.DataFrame(future_years, columns=["Year"]))

    result = {"Country": country}
    for yr, pred in zip(future_years, future_preds):
        result[str(yr)] = pred
    results.append(result)

pred_df = pd.DataFrame(results)
print("  ✔ Predictions complete\n")

max_row = pred_df.loc[pred_df["2050"].idxmax()]
min_row = pred_df.loc[pred_df["2050"].idxmin()]

print("  Predicted Extremes by 2050:")
divider()
print(f"  {'Most Affected':<25} {max_row['Country']:<28} {max_row['2050']:>7.4f} °C")
print(f"  {'Least Affected':<25} {min_row['Country']:<28} {min_row['2050']:>7.4f} °C")
divider()

baseline            = df[["Country Name", "1970"]].rename(columns={"1970": "baseline"})
merged              = pred_df.merge(baseline, left_on="Country", right_on="Country Name")
merged["rise_2050"] = merged["2050"] - merged["baseline"]
countries_over_2_2050 = merged[merged["rise_2050"] > 2]["Country"].tolist()

print(f"\n  Countries predicted with >2°C rise by 2050:")
divider()
print(f"  Total : {len(countries_over_2_2050)}")
for i, c in enumerate(countries_over_2_2050, 1):
    print(f"  {i:>3}. {c}")
divider()

mean_per_year = pred_df[[str(y) for y in future_years]].mean()
print("\n  Global Predicted Yearly Mean Temperature Change (2022–2050):")
divider()
print(f"  {'Year':<8} {'Predicted Avg Temp Change (°C)':>32}")
divider()
for yr, val in mean_per_year.items():
    print(f"  {yr:<8} {val:>32.4f}")
divider()

nepal_row = pred_df.loc[pred_df["Country"] == "Nepal", "2050"]
if not nepal_row.empty:
    print(f"\n  Nepal's Predicted Temperature Change in 2050 : {nepal_row.values[0]:.4f} °C")
else:
    print("\n  Nepal not found in dataset.")
divider()


# ─────────────────────────────────────────────────────────────
# 7. HISTORICAL + PREDICTED TREND CHART
# ─────────────────────────────────────────────────────────────
section("7. HISTORICAL + FULL REGRESSION LINE")

# Historical global mean
historical_mean = df[[str(y) for y in range(1970, 2022)]].mean()

# Convert historical years to integers
historical_years = historical_mean.index.astype(int)

# Create full year range
all_years = np.arange(1970, 2051).reshape(-1, 1)

# Predict full regression line
full_predictions = model.predict(pd.DataFrame(all_years, columns=["Year"]))

plt.figure(figsize=(18, 6))

# Historical data
plt.plot(historical_years, historical_mean.values,
         color="blue", marker="o", markersize=3,
         label="Historical Data (1970–2021)")

# Regression line
plt.plot(all_years.flatten(), full_predictions,
         color="red", linewidth=2,
         label="Linear Regression Trend (1970–2050)")

plt.axhline(2, color="gray", linestyle="--", label="2°C Threshold")

plt.title("Global Mean Temperature Change (1970–2050)")
plt.xlabel("Year")
plt.ylabel("Mean Temperature Change (°C)")
plt.legend()
plt.grid(True)
plt.xlim(1970, 2050)
plt.xticks(np.arange(1970, 2051, 5))
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 8. INTERACTIVE YEAR PREDICTION
# ─────────────────────────────────────────────────────────────
section("8. INTERACTIVE PREDICTION")

try:
    year_input = float(input("  Enter a year to predict global temperature change: "))
    pred_val   = model.predict(pd.DataFrame([[year_input]], columns=["Year"]))[0]

    print()
    divider()
    print(f"  Year                          : {int(year_input)}")
    print(f"  Predicted Temperature Change  : {pred_val:.4f} °C")
    divider()

except ValueError:
    print(" Invalid input. Please enter a numeric year.")