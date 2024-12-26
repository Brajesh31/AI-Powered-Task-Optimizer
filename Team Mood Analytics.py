import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load JSON data
with open("/Users/apple/Downloads/AI-Powered Task Optimizer/mood_tracking.json", "r") as file:
    data = json.load(file)

# Flatten the JSON structure
records = []
for employee_id, moods in data.items():
    for mood_entry in moods:
        records.append({
            "employee_id": employee_id,
            "mood": mood_entry["mood"],
            "timestamp": mood_entry["timestamp"]
        })

# Convert to DataFrame
df = pd.DataFrame(records)

# Map mood to a numerical score for aggregation
mood_to_score = {"happy": 9, "neutral": 6, "stressed": 4, "sad": 3, "angry": 2}
df["mood_score"] = df["mood"].map(mood_to_score)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract date for daily aggregation
df["date"] = df["timestamp"].dt.date

# Mock team assignment for demonstration
df["team_id"] = df["employee_id"].apply(lambda x: int(x[-1]) % 3 + 1)  # Example: assign to 3 teams

# Aggregate mood scores by team and date
team_mood = df.groupby(["team_id", "date"]).agg({"mood_score": "mean"}).reset_index()

# Visualization
plt.figure(figsize=(10, 6))

for team in team_mood["team_id"].unique():
    team_data = team_mood[team_mood["team_id"] == team]
    plt.plot(team_data["date"], team_data["mood_score"], marker="o", label=f"Team {team}")

plt.title("Team Mood Analytics")
plt.xlabel("Date")
plt.ylabel("Average Mood Score")
plt.legend()
plt.grid()
plt.tight_layout()

# Save or show the plot
plt.savefig("team_mood_trends.png")
plt.show()

# Highlight teams with low morale
threshold = 6  # Define a low morale threshold
low_morale_teams = team_mood[team_mood["mood_score"] < threshold]
print("Teams with low morale:")
print(low_morale_teams)
