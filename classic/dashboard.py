import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Fast Food A/B Testing Dashboard")

df = pd.read_csv("datasets/WA_Marketing-Campaign.csv")

st.subheader("Weekly Sales Trend by Promotion")

weekly_avg = df.groupby(["week", "Promotion"])["SalesInThousands"].mean().reset_index()

fig1, ax1 = plt.subplots()
sns.lineplot(data=weekly_avg, x="week", y="SalesInThousands", hue="Promotion", marker="o", ax=ax1)
ax1.set_title("Average Weekly Sales")
ax1.set_ylabel("Sales in Thousands")
ax1.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig1)

st.subheader("Promotion Group Comparison with 95% Confidence Intervals")

def calc_ci(data):
    mean = data.mean()
    sem = data.sem()
    margin = 1.96 * sem
    return pd.Series({"mean": mean, "ci_low": mean - margin, "ci_high": mean + margin})

summary = df.groupby("Promotion")["SalesInThousands"].apply(calc_ci).unstack().reset_index()

summary["error"] = summary["mean"] - summary["ci_low"]

fig2, ax2 = plt.subplots()
ax2.bar(summary["Promotion"], summary["mean"], yerr=summary["error"], capsize=8, color='skyblue', edgecolor='black')
ax2.set_title("Mean Sales by Promotion Group with 95% CI")
ax2.set_ylabel("Sales in Thousands")
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
st.pyplot(fig2)