import mediapipe as mp
import cv2
import tempfile
import subprocess
import time
import pandas as pd

def convert_to_h264(input_path, output_path):
    command = f"ffmpeg -i {input_path} -vcodec libx264 {output_path}"
    subprocess.run(command, shell=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def pred_in_video(video_file):
    if video_file:
        cap = cv2.VideoCapture(video_file)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        # out_file = temp_file.name  # Get temp file path

        # out = cv2.VideoWriter(
        #                     out_file,
        #                     cv2.VideoWriter_fourcc(*"mp4v"),
        #                     fps,
        #                     (frame_width, frame_height)
        #                     )
        
        data_dict = {'time':[]}
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # timestamp = frame_number / fps
            frame_number += 1

            # Convert frame to RGB (MediaPipe requires RGB input)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = pose.process(rgb_frame)
            
            # Draw landmarks and Record if detected
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, 
                                                          results.pose_landmarks, 
                                                          mp_pose.POSE_CONNECTIONS)
                result_landmark = results.pose_landmarks.landmark

                data_dict['time'].append(frame_number / fps)
                
                for i in range(len(result_landmark)):
                    column_x = f'keypoint_{i}_x'
                    column_y = f'keypoint_{i}_y'
                    column_z = f'keypoint_{i}_z'
                    column_vi = f'keypoint_{i}_visibility'

                    if column_x not in data_dict:
                        data_dict[column_x] = []
                        data_dict[column_y] = []
                        data_dict[column_z] = []
                        data_dict[column_vi] = []
                    
                    # data_dict['time'].append(frame_number / fps)
                    data_dict[column_x].append(result_landmark[i].x)
                    data_dict[column_y].append(result_landmark[i].y)
                    data_dict[column_z].append(result_landmark[i].z)
                    data_dict[column_vi].append(result_landmark[i].visibility)

            # Save frame to output video
            # out.write(frame)

    cap.release()
    # out.release()

    df = pd.DataFrame(data_dict)
    # converted_file = out_file.replace(".mp4", "_converted.mp4")
    # convert_to_h264(out_file, converted_file)

    # time.sleep(1)
    # converted_file,
    return df