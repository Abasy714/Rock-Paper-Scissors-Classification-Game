import cv2
from ultralytics import YOLO

model = YOLO('best .pt')  

cap = cv2.VideoCapture(0)  
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting camera feed...")

gesture_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}

def determine_winner(gesture1, gesture2):
    if gesture1 == gesture2:
        return "Draw"
    if (gesture1 == 'Rock' and gesture2 == 'Scissors') or \
       (gesture1 == 'Scissors' and gesture2 == 'Paper') or \
       (gesture1 == 'Paper' and gesture2 == 'Rock'):
        return "Player 1 Wins"
    return "Player 2 Wins"

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Failed to capture image.")
        break

 
    results = model(frame, stream=True)  

    player1_gesture = None
    player2_gesture = None

    for result in results:
        boxes = result.boxes  
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = gesture_map.get(cls, "Unknown")

            center_x = (x1 + x2) // 2
            frame_center = frame.shape[1] // 2

            if center_x < frame_center:
                player1_gesture = label
                player_text = f"Player 1: {label}"
                cv2.putText(frame, player_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                player2_gesture = label
                player_text = f"Player 2: {label}"
                cv2.putText(frame, player_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if player1_gesture and player2_gesture:
        winner_text = determine_winner(player1_gesture, player2_gesture)
        cv2.putText(frame, winner_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Rock Paper Scissors Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
