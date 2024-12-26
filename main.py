import cv2
import numpy as np
import librosa
import torch
from transformers import pipeline
import sounddevice as sd
import matplotlib.pyplot as plt
import random
import time
import atexit
import sys

# ===== TEXT ANALYSIS =====
# Specify the model name and revision
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
revision = "714eb0f"

# Initialize the pipeline with the specified model and revision
text_analyzer = pipeline("sentiment-analysis", model=model_name, revision=revision)

def analyze_text_emotion(text):
    """
    Analyze the emotion from the given text.
    """
    result = text_analyzer(text)
    return result[0]['label'], result[0]['score']

# ===== FACIAL EXPRESSION ANALYSIS =====
# Load a pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_facial_expression(frame):
    """
    Analyze facial expressions using pre-trained emotion detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]  # Simplified emotion list

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        emotion = np.random.choice(emotions)  # Replace with model prediction
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return emotion
    return "No Face Detected"

# ===== SPEECH EMOTION DETECTION =====
def analyze_speech_emotion(duration=3, sampling_rate=16000):
    """
    Analyze speech emotion from microphone input.
    """
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete!")

    # Extract features for emotion detection
    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=sampling_rate, n_mfcc=40)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    emotion = np.random.choice(["Neutral", "Happy", "Sad", "Angry", "Fearful"])
    return emotion

# ===== TASK RECOMMENDATIONS =====
# Predefined tasks categorized by mood
task_recommendations = {
    "happy": [
        "Lead a brainstorming session",
        "Collaborate on a creative project",
        "Mentor a colleague",
        "Work on a challenging task that excites you"
    ],
    "sad": [
        "Take a short break and relax",
        "Focus on light, repetitive tasks like organizing emails",
        "Review previously completed tasks for confidence",
        "Engage in mindfulness activities or journaling"
    ],
    "angry": [
        "Step away for a moment and practice deep breathing",
        "Engage in physical tasks like organizing the workspace",
        "Work on analytical tasks that require focus",
        "Draft notes or outlines for future projects"
    ],
    "neutral": [
        "Continue working on your scheduled tasks",
        "Review and update project documentation",
        "Learn a new skill or explore training modules",
        "Check in with colleagues or team members"
    ],
    "stressed": [
        "Prioritize tasks and focus on the most critical ones",
        "Delegate tasks if possible",
        "Attend a stress management workshop or session",
        "Take regular breaks during the workday"
    ],
    "excited": [
        "Start a high-priority project",
        "Take on leadership roles for team activities",
        "Share your ideas with the team in a meeting",
        "Work on innovative or creative assignments"
    ]
}

def recommend_task(emotion):
    if emotion.lower() in task_recommendations:
        tasks = task_recommendations[emotion.lower()]
        return random.choice(tasks)
    else:
        return "Emotion not recognized. Please ensure the emotion is valid."

# ===== REAL-TIME SYSTEM INTEGRATION =====
def real_time_emotion_detection():
    """
    Real-time emotion detection system with a 60-second limit.
    """
    print("Launching Real-Time Emotion Detection...")
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    detected_emotions = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Facial Expression Analysis
            emotion_face = analyze_facial_expression(frame)
            if emotion_face != "No Face Detected":
                detected_emotions.append(emotion_face)

            # Display live video with facial emotion overlay
            cv2.imshow("Emotion Detection", frame)

            # Stop after 60 seconds
            if time.time() - start_time > 60:
                break

            # Keyboard interrupt to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Determine most frequent emotion from detected emotions
        if detected_emotions:
            final_facial_emotion = max(set(detected_emotions), key=detected_emotions.count)
        else:
            final_facial_emotion = "No Face Detected"

        # Text Analysis
        text_input = input("Enter text to analyze emotion: ")
        emotion_text, confidence_text = analyze_text_emotion(text_input)
        print(f"Text Emotion: {emotion_text} (Confidence: {confidence_text:.2f})")

        # Speech Analysis
        emotion_speech = analyze_speech_emotion()
        print(f"Speech Emotion: {emotion_speech}")

        # Task Recommendation
        detected_emotion = emotion_text.lower()  # Using text emotion for task recommendation
        recommended_task = recommend_task(detected_emotion)

        print(f"\nFinal Emotion Analysis Results:")
        print(f"Facial Expression: {final_facial_emotion}")
        print(f"Text Analysis: {emotion_text}")
        print(f"Speech Emotion: {emotion_speech}")
        print(f"Recommended Task: {recommended_task}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cleanup()

def cleanup():
    """
    Cleanup function to be called on exit.
    """
    print("Cleaning up resources...")
    # Add any resource cleanup code here
    cv2.destroyAllWindows()
    sd.stop()

# Register the cleanup function to be called on exit
atexit.register(cleanup)

# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    try:
        real_time_emotion_detection()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        cleanup()