import pandas as pd
from cryptography.fernet import Fernet
import hashlib

# Sample employee data (replace with real data)
data = [
    {"employee_id": 1, "name": "Akash Jha", "email": "akashjha@gmail.com", "mood_score": 7},
    {"employee_id": 2, "name": "Deepak K", "email": "deepakk@gmail.com", "mood_score": 6},
    {"employee_id": 3, "name": "Dharmendra Jha", "email": "dharmendra@gmail.com", "mood_score": 5},
    {"employee_id": 4, "name": "Abhishek Singh", "email": "abhisheksingh@gmail.com", "mood_score": 8},
]

# Convert data to DataFrame
df = pd.DataFrame(data)

# Generate encryption key
key = Fernet.generate_key()
cipher = Fernet(key)

# Function to anonymize sensitive fields
def anonymize_data(row):
    row["name"] = hashlib.sha256(row["name"].encode()).hexdigest()  # Hash the name
    row["email"] = cipher.encrypt(row["email"].encode()).decode()  # Encrypt the email
    return row

# Apply anonymization
df = df.apply(anonymize_data, axis=1)

# Save encryption key securely (for demonstration purposes, printing here is NOT secure)
print("Encryption key (store this securely!):", key.decode())

# Save anonymized data to secure storage
secure_file = "anonymized_data.csv"
df.to_csv(secure_file, index=False)

# Read and decrypt data (example for authorized access)
def decrypt_email(encrypted_email):
    return cipher.decrypt(encrypted_email.encode()).decode()

# Test decryption
print("Decrypted email example:", decrypt_email(df.iloc[0]["email"]))

# Output anonymized data
print("Anonymized Data:")
print(df)
