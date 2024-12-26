import json
from datetime import datetime, timedelta

# Mood tracking database (JSON file for simplicity)
MOOD_DATABASE_FILE = "mood_tracking.json"

# HR notification function (simulated)
def notify_hr(employee_id, mood):
    print(f"ALERT: Employee ID: {employee_id} has shown prolonged {mood}!")
    print("HR or Manager has been notified to take action.")

# Function to load mood data from the database
def load_mood_data():
    try:
        with open(MOOD_DATABASE_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Function to check for prolonged stress or disengagement
def check_prolonged_mood(employee_id, mood="stressed", threshold_days=3):
    data = load_mood_data()

    if employee_id not in data:
        print(f"No mood data available for Employee ID: {employee_id}")
        return
    
    # Get mood entries for the employee
    mood_entries = data[employee_id]
    
    # Filter entries for the specific mood
    prolonged_moods = [
        entry for entry in mood_entries
        if entry["mood"] == mood and
           datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") >= datetime.now() - timedelta(days=threshold_days)
    ]
    
    # Notify if the mood has been consistent for the threshold period
    if len(prolonged_moods) >= threshold_days:
        notify_hr(employee_id, mood)
    else:
        print(f"Employee ID: {employee_id} does not show prolonged {mood}.")

# Function to simulate recording a mood (reuse from previous phases)
def record_mood(employee_id, mood):
    data = load_mood_data()
    
    if employee_id not in data:
        data[employee_id] = []
    
    entry = {
        "mood": mood,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    data[employee_id].append(entry)
    save_mood_data(data)
    print(f"Mood recorded for Employee ID: {employee_id}")

# Function to save mood data to the database
def save_mood_data(data):
    with open(MOOD_DATABASE_FILE, "w") as file:
        json.dump(data, file, indent=4)

# Example usage
if __name__ == "__main__":
    # Simulate mood recordings
    record_mood("E001", "stressed")
    record_mood("E001", "stressed")
    record_mood("E001", "stressed")
    record_mood("E002", "happy")
    
    # Check for prolonged stress
    check_prolonged_mood("E001", mood="stressed", threshold_days=3)
    check_prolonged_mood("E002", mood="stressed", threshold_days=3)
