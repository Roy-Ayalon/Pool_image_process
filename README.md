# Pool Image Assist

![WhatsApp Image 2025-03-10 at 08 19 18](https://github.com/user-attachments/assets/a76efd8d-d7e5-4c4a-bd35-ace594f7dbbb)

## Overview
Pool Image Assist is an interactive system designed to help beginner 8-ball pool players improve their accuracy and understanding of the game's physics. The system uses computer vision techniques to track the cue, balls, and table, providing real-time visual guidance to the player.

By analyzing the player's shot alignment, the system displays the predicted trajectory of the cue ball and the targeted ball, assisting players in making more informed shots.

## Features
- **Real-time ball and cue detection**: Identifies the cue, cue ball, and object balls using computer vision.
- **Trajectory prediction**: Estimates and visualizes the path of the cue ball and the potential movement of the targeted ball.
- **Live feedback**: Displays the predicted shot on-screen to help players adjust their aim.
- **Score tracking**: Monitors the game by detecting balls that have been pocketed.

## System Components
### Hardware Requirements
- Overhead camera mounted above the pool table
- Computer with OpenCV and Python support

### Software Components
The system consists of several Python modules:
- `detect_board.py` - Identifies the table boundaries and playing area.
- `detect_balls.py` - Detects and classifies the balls based on color and position.
- `detect_stick.py` - Tracks the cue stick to determine shot direction.
- `four_lines.py` - Identifies the major edges of the table.
- `trajectory.py` - Predicts and visualizes ball movement after impact.
- `is_moving.py` - Detects whether balls are in motion.
- `app.py` - Main application integrating all modules.

## Algorithm Overview
The system utilizes several image processing techniques:
1. **Table Detection**
   - Converts frames to the LAB color space and extracts the green table area.
   - Uses contour detection to outline the table boundaries.

2. **Ball Detection**
   - Converts frames to grayscale and applies the Hough Circle Transform to detect balls.
   - Classifies balls based on their HSV color representation.

3. **Cue Stick Detection**
   - Converts frames to CYMK color space and extracts the cue stick.
   - Uses Hough Line Transform to detect the primary direction of the cue.

4. **Trajectory Prediction**
   - Determines if the cue stick is aligned with the cue ball.
   - Simulates the projected movement of the cue ball and its impact on other objects (balls, pockets, walls).

5. **Game Tracking**
   - Monitors which balls remain on the table.
   - Awards points based on detected pockets.

## Installation & Usage
### Prerequisites
- Python 3.11
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Running the Application
1. Set up the overhead camera to capture the pool table.
2. Run the application with:
   ```sh
   python app.py
   ```
3. Aim the cue and observe real-time guidance on-screen.

## Challenges & Future Improvements
### Challenges Faced
- **Lighting variations**: Adjusting to different illumination conditions.
- **Real-time processing**: Optimizing image processing speed for smooth gameplay.
- **Object occlusions**: Handling cases where the cue stick or balls overlap.

### Future Improvements
- Implement AI-based shot recommendations.
- Improve robustness against varying lighting conditions.
- Enhance tracking for moving objects and dynamic repositioning.

## Demonstration


https://github.com/user-attachments/assets/3ee8c6c4-1594-45d5-bc05-4a03ad9143c7



## Contributors
- **Bar Binyamin Varsulker**
- **Roy Ayalon**
- **Michael Tatarijitzky**
- **Adir Lastfoguel**
  
![WhatsApp Image 2025-03-10 at 11 32 28](https://github.com/user-attachments/assets/2e230500-bab1-44d1-aced-4c686494d8a4)

---
This project serves as a proof of concept for using computer vision in sports training, providing real-time feedback and interactive learning for pool players.

