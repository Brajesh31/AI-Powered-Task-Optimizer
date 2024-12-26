import json
from datetime import datetime

# Mood tracking database (JSON file for simplicity)
MOOD_DATABASE_FILE = "mood_tracking.json"

# Function to load mood data from the database
def load_mood_data():
    try:
        with open(MOOD_DATABASE_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Function to save mood data to the database
def save_mood_data(data):
    with open(MOOD_DATABASE_FILE, "w") as file:
        json.dump(data, file, indent=4)

# Function to record a mood entry for an employee
def record_mood(employee_id, mood):
    data = load_mood_data()
    
    # Add new employee entry if not already present
    if employee_id not in data:
        data[employee_id] = []
    
    # Record the current mood with timestamp
    entry = {
        "mood": mood,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    data[employee_id].append(entry)
    
    # Save updated data
    save_mood_data(data)
    print(f"Mood recorded for Employee ID: {employee_id}")

# Function to analyze mood trends for an employee
def analyze_mood_trends(employee_id):
    data = load_mood_data()
    
    if employee_id not in data:
        print(f"No mood data available for Employee ID: {employee_id}")
        return
    
    moods = [entry["mood"] for entry in data[employee_id]]
    mood_counts = {mood: moods.count(mood) for mood in set(moods)}
    
    print(f"Mood Trends for Employee ID: {employee_id}")
    for mood, count in mood_counts.items():
        print(f"- {mood}: {count} times")

# Example usage
if __name__ == "__main__":
    # Record moods for employees
    record_mood("E001", "happy")
    record_mood("E001", "neutral")
    record_mood("E001", "stressed")
    record_mood("E002", "sad")
    record_mood("E002", "happy")
    
    # Analyze mood trends
    analyze_mood_trends("E001")
    analyze_mood_trends("E002")
