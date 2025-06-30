import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import time
from deepface import DeepFace
import traceback

class EmotionConfidenceApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video capture device")
        
       
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise Exception("Error loading face cascade classifier")
        
       
        self.confidence_history = deque(maxlen=30)  
        self.emotion_history = deque(maxlen=10)     
        self.start_time = time.time()
        self.frame_count = 0
        self.frame = None
        
        
        self.emotion_confidence_scores = {
            'angry': 30,     
            'disgust': 20,   
            'fear': 25,      
            'happy': 90,     
            'sad': 40,       
            'surprise': 70,  
            'neutral': 60    
        }
        
       
        cv2.namedWindow('Emotion Confidence Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Confidence Monitor', 1200, 720)
        
        
        self.fig = plt.figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 30)
        self.ax.set_title('Confidence Score Trend')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Confidence Score (%)')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        
        
        self.canvas = FigureCanvasAgg(self.fig)
        
        
        self.confidence_history.append(0)
        self.current_confidence = 0
        self.current_emotion = "unknown"
        self.current_emotion_scores = {}

    def calculate_confidence(self, face_roi, face_coords):
        """Calculate overall confidence score based on emotion and face attributes."""
        x, y, w, h = face_coords
        
        try:
            
            face_img = face_roi.copy()
            
            
            if face_img.size == 0 or face_img is None:
                print("Invalid face ROI")
                return 0, "unknown", {}
                
            
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                face_img = cv2.resize(face_img, (48, 48))
            
            
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            if not analysis or len(analysis) == 0:
                print("No analysis results returned")
                return 0, "unknown", {}
                
            
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_scores = analysis[0]['emotion']
            dominant_score = emotion_scores[dominant_emotion]
            
            
            if dominant_emotion not in self.emotion_confidence_scores:
                print(f"Unknown emotion detected: {dominant_emotion}")
                dominant_emotion = "neutral"
            
            
            self.emotion_history.append(dominant_emotion)
            
            
            emotion_base_score = self.emotion_confidence_scores[dominant_emotion]
            emotion_confidence = 0.4 * emotion_base_score
            
            
            certainty_factor = 0.2 * dominant_score
            
            
            if self.frame is not None:
                frame_height, frame_width = self.frame.shape[:2]
                size_factor = (w * h) / (frame_width * frame_height) * 100
                size_score = min(size_factor * 3, 15)
            else:
                size_score = 0
            
            
            if self.frame is not None:
                frame_height, frame_width = self.frame.shape[:2]
                center_x = x + w/2
                center_y = y + h/2
                center_offset_x = abs(center_x - frame_width/2) / (frame_width/2)
                center_offset_y = abs(center_y - frame_height/2) / (frame_height/2)
                position_score = 10 * (1 - (center_offset_x + center_offset_y) / 2)
            else:
                position_score = 0
            
            
            if len(self.emotion_history) >= 3:
               
                emotion_count = self.emotion_history.count(dominant_emotion)
                stability_ratio = emotion_count / len(self.emotion_history)
                stability_score = 15 * stability_ratio
            else:
                stability_score = 0
            
            
            total_confidence = min(100, emotion_confidence + certainty_factor + size_score + position_score + stability_score)
            
            return total_confidence, dominant_emotion, emotion_scores
            
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            traceback.print_exc()
            return 0, "unknown", {}

    def update_plot(self):
        """Update the confidence plot."""
        try:
            x = list(range(len(self.confidence_history)))
            y = list(self.confidence_history)
            
            self.line.set_data(x, y)
            
            
            self.canvas.draw()
            plot_img = np.array(self.canvas.renderer.buffer_rgba())
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
            
            return plot_img
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            
            return np.ones((300, 500, 3), dtype=np.uint8) * 240

    def create_emotion_bar(self, emotion_scores, width=400, height=150):
        """Create a horizontal bar chart of emotion scores."""
        try:
            
            bar_chart = np.ones((height, width, 3), dtype=np.uint8) * 240
            
            
            if not emotion_scores:
                cv2.putText(bar_chart, "No emotions detected", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
                return bar_chart
            
            
            colors = {
                'angry': (0, 0, 200),      
                'disgust': (0, 130, 0),   
                'fear': (130, 0, 130),    
                'happy': (0, 215, 255),    
                'sad': (139, 69, 19),     
                'surprise': (255, 140, 0), 
                'neutral': (128, 128, 128) 
            }
            
            
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            
            max_bar_width = width - 150
            bar_height = 20
            start_y = 10
            
            for i, (emotion, score) in enumerate(sorted_emotions):
               
                cv2.putText(bar_chart, f"{emotion}", (10, start_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                
                bar_width = int(max_bar_width * score / 100)
                cv2.rectangle(bar_chart, 
                             (100, start_y), 
                             (100 + bar_width, start_y + bar_height), 
                             colors.get(emotion, (0, 0, 0)), 
                             -1)
                
               
                cv2.putText(bar_chart, f"{score:.1f}%", (100 + bar_width + 5, start_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                start_y += bar_height + 10
            
            return bar_chart
        except Exception as e:
            print(f"Error creating emotion bar: {str(e)}")
            return np.ones((height, width, 3), dtype=np.uint8) * 240

    def create_dashboard(self):
        """Create the dashboard display with video, confidence metrics and emotion."""
        try:
            dashboard = np.ones((720, 1200, 3), dtype=np.uint8) * 240
            
            if self.frame is not None:
                video_display = cv2.resize(self.frame, (600, 450))
                dashboard[50:500, 50:650] = video_display
            
            
            plot_img = self.update_plot()
            plot_h, plot_w = plot_img.shape[:2]
            dashboard[50:50+plot_h, 700:700+plot_w] = plot_img
            
            
            emotion_bars = self.create_emotion_bar(self.current_emotion_scores)
            e_h, e_w = emotion_bars.shape[:2]
            dashboard[350:350+e_h, 700:700+e_w] = emotion_bars
            
            
            cv2.rectangle(dashboard, (700, 550), (1100, 600), (220, 220, 220), -1)
           
            fill_width = int(400 * (self.current_confidence / 100))
            color = self.get_color_for_confidence(self.current_confidence)
            cv2.rectangle(dashboard, (700, 550), (700 + fill_width, 600), color, -1)
            
            
            cv2.putText(dashboard, f"Overall Confidence: {self.current_confidence:.1f}%", 
                       (700, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            
            
            cv2.putText(dashboard, f"Primary Emotion: {self.current_emotion}", 
                       (700, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            
            
            cv2.putText(dashboard, "Confidence Factors:", 
                       (50, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            
            
            if self.current_emotion != "unknown" and self.current_emotion in self.emotion_confidence_scores:
                emotion_base = self.emotion_confidence_scores[self.current_emotion]
                emotion_detect = self.current_emotion_scores.get(self.current_emotion, 0)
                
                if len(self.emotion_history) >= 3:
                    emotion_count = self.emotion_history.count(self.current_emotion)
                    stability_ratio = emotion_count / len(self.emotion_history)
                    if stability_ratio > 0.7:
                        stability_text = "High"
                    elif stability_ratio > 0.4:
                        stability_text = "Moderate"
                    else:
                        stability_text = "Low"
                else:
                    stability_text = "Insufficient data"
                
                factors = [
                    f"Emotion Type: {emotion_base}/100 base value",
                    f"Emotion Certainty: {emotion_detect:.1f}% detection confidence",
                    f"Face Position & Size: Weighted for optimal framing",
                    f"Emotional Stability: {stability_text}"
                ]
            else:
                factors = [
                    "• No face detected or emotion recognized",
                    "• Position camera to capture your face clearly",
                    "• Ensure adequate lighting",
                    "• Try different facial expressions"
                ]
            
            y_pos = 570
            for factor in factors:
                cv2.putText(dashboard, factor, 
                           (60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
                y_pos += 30
            
            
            elapsed_time = max(1, time.time() - self.start_time)
            fps = self.frame_count / elapsed_time
            cv2.putText(dashboard, f"FPS: {fps:.1f}", 
                       (1100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            
            cv2.putText(dashboard, "Emotion-Based Confidence Analysis", 
                       (400, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (30, 30, 30), 2)
            
           
            if self.current_emotion == "unknown":
                status_msg = "Status: Waiting for face detection..."
            else:
                status_msg = "Status: Analyzing emotions"
                
            cv2.putText(dashboard, status_msg, 
                       (50, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
            
            return dashboard
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            
            error_dashboard = np.ones((720, 1200, 3), dtype=np.uint8) * 240
            cv2.putText(error_dashboard, "Error creating dashboard", 
                       (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_dashboard
    
    def get_color_for_confidence(self, confidence):
        """Return a color based on confidence level (from red to green)."""
        if confidence < 30:
            return (0, 0, 180)  
        elif confidence < 60:
            return (0, 180, 180)  
        else:
            return (0, 180, 0)  

    def run(self):
        """Main application loop."""
        print("Starting Emotion Confidence App...")
        print("Press 'q' to quit")
        
        while True:
            try:
                ret, self.frame = self.cap.read()
                if not ret or self.frame is None:
                    print("Failed to grab frame, trying again...")
                    
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(0)
                    continue
                
               
                self.frame = cv2.flip(self.frame, 1)
                
               
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                
               
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                   
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face
                    
                    
                    face_roi = self.frame[y:y+h, x:x+w].copy()
                    
                   
                    confidence, emotion, emotion_scores = self.calculate_confidence(face_roi, (x, y, w, h))
                    self.current_confidence = confidence
                    self.current_emotion = emotion
                    self.current_emotion_scores = emotion_scores
                    self.confidence_history.append(confidence)
                    
                    
                    cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    
                    cv2.putText(self.frame, f"{emotion.capitalize()}", (x, y-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(self.frame, f"Confidence: {confidence:.1f}%", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                   
                    if self.confidence_history[-1] != 0:
                        self.confidence_history.append(0)
                    self.current_confidence = 0
                    self.current_emotion = "unknown"
                    self.current_emotion_scores = {}
                
               
                dashboard = self.create_dashboard()
                
               
                cv2.imshow('Emotion Confidence Monitor', dashboard)
                
               
                self.frame_count += 1
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                traceback.print_exc()
                time.sleep(0.5)  
        print("Shutting down...")
        self.cap.release()
        cv2.destroyAllWindows()
        plt.close(self.fig)

def main():
    """Main function to run the application."""
    print("Initializing Emotion Confidence Application...")
    try:
        app = EmotionConfidenceApp()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        traceback.print_exc()
        cv2.destroyAllWindows()
        plt.close('all')

if __name__ == "__main__":
    main()
