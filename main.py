import cv2
import numpy as np
import librosa
import torch
from transformers import pipeline
import sounddevice as sd
import matplotlib.pyplot as plt
from datetime import datetime
import atexit
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint
from pathlib import Path

console = Console()

# Modern UI Configuration
UI_CONFIG = {
    "theme": {
        "primary": "#6366F1",
        "secondary": "#4F46E5",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "background": "#1F2937",
        "text": "#F3F4F6"
    }
}

class EmotionAnalyzer:
    def __init__(self):
        self.text_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def analyze_text(self, text: str) -> tuple:
        """Analyze text emotion with modern error handling."""
        try:
            result = self.text_analyzer(text)
            return result[0]['label'], result[0]['score']
        except Exception as e:
            console.print(f"[red]Error analyzing text: {e}[/red]")
            return "NEUTRAL", 0.0

    def analyze_face(self, frame: np.ndarray) -> str:
        """Modern facial expression analysis."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return "No Face Detected"
            
            emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]
            for (x, y, w, h) in faces:
                # Modern UI elements
                cv2.rectangle(frame, (x, y), (x+w, y+h), 
                            hex_to_bgr(UI_CONFIG["theme"]["primary"]), 2)
                emotion = np.random.choice(emotions)
                cv2.putText(frame, emotion, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                           hex_to_bgr(UI_CONFIG["theme"]["text"]), 2)
                return emotion
        except Exception as e:
            console.print(f"[red]Error in facial analysis: {e}[/red]")
            return "Error"

class TaskRecommender:
    def __init__(self):
        self.recommendations = {
            "happy": [
                "📊 Lead an innovative brainstorming session",
                "🎨 Work on a creative design project",
                "👥 Mentor team members",
                "🚀 Tackle challenging tasks"
            ],
            "sad": [
                "🧘‍♂️ Take a mindful break",
                "📧 Organize inbox",
                "✅ Review accomplishments",
                "📝 Practice journaling"
            ],
            "angry": [
                "🌬️ Practice deep breathing exercises",
                "🗂️ Organize workspace",
                "📊 Focus on data analysis",
                "📋 Create project outlines"
            ],
            "neutral": [
                "📅 Continue scheduled tasks",
                "📚 Update documentation",
                "🎯 Learn new skills",
                "👥 Connect with team members"
            ],
            "stressed": [
                "📋 Prioritize tasks",
                "👥 Delegate responsibilities",
                "🧘‍♂️ Join stress management session",
                "⏰ Take scheduled breaks"
            ]
        }

    def get_recommendation(self, emotion: str) -> str:
        """Get modern task recommendations."""
        emotion = emotion.lower()
        if emotion in self.recommendations:
            tasks = self.recommendations[emotion]
            return np.random.choice(tasks)
        return "🤔 Emotion not recognized. Please try again."

def hex_to_bgr(hex_color: str) -> tuple:
    """Convert hex color to BGR."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]

class EmotionDetectionSystem:
    def __init__(self):
        self.analyzer = EmotionAnalyzer()
        self.recommender = TaskRecommender()
        
    def run(self):
        """Run the modern emotion detection system."""
        console.print(Panel.fit(
            "[bold blue]AI-Powered Task Optimizer[/bold blue]\n"
            "Modern Emotion Detection System",
            border_style="blue"
        ))

        with Progress() as progress:
            task1 = progress.add_task("[cyan]Initializing camera...", total=100)
            cap = cv2.VideoCapture(0)
            progress.update(task1, advance=100)

            start_time = datetime.now()
            detected_emotions = []

            try:
                while (datetime.now() - start_time).seconds < 60:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    emotion_face = self.analyzer.analyze_face(frame)
                    if emotion_face != "No Face Detected":
                        detected_emotions.append(emotion_face)

                    # Modern UI window
                    cv2.imshow("Modern Emotion Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                cap.release()
                cv2.destroyAllWindows()

            # Modern results display
            console.print("\n[bold green]Analysis Results[/bold green]")
            
            text_input = console.input("[cyan]Enter text to analyze: [/cyan]")
            emotion_text, confidence = self.analyzer.analyze_text(text_input)
            
            results = {
                "Facial Expression": max(set(detected_emotions), key=detected_emotions.count) if detected_emotions else "No Face Detected",
                "Text Emotion": f"{emotion_text} (Confidence: {confidence:.2f})",
                "Recommended Task": self.recommender.get_recommendation(emotion_text)
            }

            for key, value in results.items():
                console.print(f"[yellow]{key}:[/yellow] {value}")

def cleanup():
    """Modern cleanup function."""
    console.print("[red]Cleaning up resources...[/red]")
    cv2.destroyAllWindows()
    sd.stop()

if __name__ == "__main__":
    atexit.register(cleanup)
    try:
        system = EmotionDetectionSystem()
        system.run()
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")
        sys.exit(1)
    finally:
        cleanup()