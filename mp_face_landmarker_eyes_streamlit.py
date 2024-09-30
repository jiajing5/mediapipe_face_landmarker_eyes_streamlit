import cv2
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import streamlit as st

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

def get_eye_landmarks(face_landmarks):
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    left_eye = [face_landmarks[i] for i in LEFT_EYE]
    right_eye = [face_landmarks[i] for i in RIGHT_EYE]

    return left_eye, right_eye

def calculate_ear(eye):
    vert_1 = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
    vert_2 = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
    horz = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
    ear = (vert_1 + vert_2) / (2.0 * horz)
    return ear

def is_eye_closed(ear, threshold=0.1):
    return ear < threshold

# Streamlit setup
st.title("Eye Blink Detection with Mediapipe")
EYE_CLOSED_DURATION_THRESHOLD = st.slider("Set Eye Closed Duration Threshold (seconds)", 0.1, 5.0, 3.0)

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open video.")
    st.stop()

ear_history = deque(maxlen=10)
eye_closed_start_time = None

stframe = st.empty()  # To display the image frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(image)

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        left_eye, right_eye = get_eye_landmarks(face_landmarks)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        avg_ear = (left_ear + right_ear) / 2
        ear_history.append(avg_ear)
        smooth_ear = sum(ear_history) / len(ear_history)

        eyes_closed = is_eye_closed(smooth_ear)

        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            eye_blink_left = next((shape for shape in blendshapes if shape.category_name == "eyeBlinkLeft"), None)
            eye_blink_right = next((shape for shape in blendshapes if shape.category_name == "eyeBlinkRight"), None)
            
            if eye_blink_left and eye_blink_right:
                blink_score = (eye_blink_left.score + eye_blink_right.score) / 2
                eyes_closed = eyes_closed or (blink_score > 0.5)

        if eyes_closed:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elif time.time() - eye_closed_start_time > EYE_CLOSED_DURATION_THRESHOLD:
                eyes_closed = True
            else:
                eyes_closed = False
        else:
            eye_closed_start_time = None

        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.putText(annotated_image, f"EAR: {smooth_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Eyes: {'Closed' if eyes_closed else 'Open'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stframe.image(annotated_image, channels="RGB")
    else:
        stframe.image(frame_rgb, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
