import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time

st.set_page_config(layout="wide")

st.header('Распознавание вооруженных людей на архивных видео и прямых трансляциях с камер видеонаблюдения')

# Load the YOLOv8 model
model = YOLO('best.pt')

# # Using "with" notation
# with st.sidebar:
#     st.header('Выберите параметры')
#     conf = st.number_input('Пороговая вероятность', value=0.5, min_value=0.01, max_value=1.)

tab1, tab2 = st.tabs(['Обработка одиночного видео', 'Обработка потока с онлайн-камер'])

def get_random_numpy():
        """Return a dummy frame."""
        return np.random.randint(0, 100, size=(32, 32))

with tab1:
    
    uploaded_video = st.file_uploader('Загрузите видео для обработки', type=["mp4", "mov"])

    if uploaded_video is not None: # run only when user uploads video
        
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        cap = cv2.VideoCapture(vid)
        # cap = cv2.VideoCapture(0)

        # Create a video writer to save the annotated video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize a list to store the results
        results_list = []
        results_frames = []

        # Loop through the video frames

        viewer = st.image(get_random_numpy())

        start_time = time.time()
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True, verbose=False)

                if len(results[0].boxes) > 1:
                    
                    if results[0].boxes.conf[0].item() > 0.8 and results[0].boxes.conf[1].item() > 0.5:
                    
                        # Append the results to the list
                        results_list.append(time.time() - start_time)
                    
                        # Visualize the results on the frame
                        annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                        results_frames.append(annotated_frame)

                else:

                    annotated_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

                viewer.image(annotated_frame)

            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object, video writer, and close the display window
        end_time = time.time()
        cap.release()

        for i in range(len(results_frames)):
            with st.expander(f'Распознавание на {round(results_list[i])} секунде'):
                st.image(results_frames[i])

with tab2:

    link = st.text_input('Введите __rtsp-адрес__ потокового видео')
    # login = st.text_input('Введите логин')
    # password = st.text_input('Введите пароль')

    w = 1920
    h = 1080

    login = 'admin'
    password = 'A1234567'

    # VIEWER_WIDTH = 600

    if link != '':
    
        if '.mp4' in link:
            new_link=link
        else:
            sign=':'
            l_p=''.join([login, 
                         sign, password
                         ])
            # Найдем начальную и конечную позиции символа "@"
            start_position = link.index("rtsp://") + len("rtsp://")
            end_position = link.index("@")
            new_link = link.replace(link[start_position:end_position], l_p)
        cap = cv2.VideoCapture(new_link) 
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        
        
        
        start_time = time.time()

        viewer = st.image(get_random_numpy())

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True, verbose=False)

                if len(results[0].boxes) > 1:
                    
                    if results[0].boxes.conf[0].item() > 0.8 and results[0].boxes.conf[1].item() > 0.5:
                    
                        # Append the results to the list
                        results_l = time.time() - start_time
                    
                        # Visualize the results on the frame
                        annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                        with st.expander(f'Распознавание на {round(results_l)} секунде'):
                            st.image(annotated_frame)

                else:

                    annotated_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

                viewer.image(annotated_frame)

            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object, video writer, and close the display window
        end_time = time.time()
        cap.release()
