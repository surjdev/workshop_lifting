
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io

def draw_3d_pose_from_keypoints(keypoint_df, frame, fig_size=(6,6)):
    """
    วาดภาพท่าทางในแบบ 3 มิติจาก keypoints ของ MediaPipe และคืนค่าเป็นภาพ OpenCV (BGR)
    
    Parameters:
      keypoint_df (pd.DataFrame): DataFrame ที่มีข้อมูล keypoint โดยมีคอลัมน์:
          'time' และ สำหรับแต่ละ keypoint i (0-32):
          'keypoint_{i}_x', 'keypoint_{i}_y', 'keypoint_{i}_z', และ 'keypoint_{i}_visibility'
      frame: สามารถเป็น:
          - index (int) ที่ใช้เลือกแถวจาก keypoint_df
          - pandas Series (แทนข้อมูลเฟรมเดียว)
          - 1-row DataFrame (จะถูกแปลงเป็น Series)
      fig_size (tuple): ขนาดของ Matplotlib figure
          
    Returns:
      img (np.ndarray): ภาพผลลัพธ์ในรูปแบบ OpenCV BGR ที่ได้จากการวาด 3D pose
    """
    # จัดการกับ input ที่เป็น frame หลายรูปแบบ
    if isinstance(frame, int):
        row_data = keypoint_df.iloc[frame]
    elif isinstance(frame, pd.DataFrame):
        if frame.shape[0] == 1:
            row_data = frame.iloc[0]
        else:
            raise ValueError("DataFrame ที่ส่งเข้ามาต้องมีเพียง 1 แถว")
    elif isinstance(frame, pd.Series):
        row_data = frame
    else:
        raise TypeError("frame ต้องเป็น int, Series, หรือ 1-row DataFrame")

    time_val = row_data['time']
    num_keypoints = 33  # MediaPipe Pose มี 33 keypoints
    kp = np.zeros((num_keypoints, 3))
    for i in range(num_keypoints):
        kp[i, 0] = row_data[f'keypoint_{i}_x']
        kp[i, 1] = row_data[f'keypoint_{i}_y']
        kp[i, 2] = row_data[f'keypoint_{i}_z']

    # กำหนด keypoint สำหรับไหล่และข้อเท้า (ตาม convention ของ MediaPipe)
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER  = 11
    RIGHT_ANKLE    = 28
    LEFT_ANKLE     = 27

    # สร้าง figure 3D
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # วาด keypoints ทั้งหมด (สีน้ำเงิน)
    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c='b', s=20, label='Keypoints')

    # วาดเส้นเชื่อมระหว่างไหล่และระหว่างข้อเท้า
    ax.plot([kp[RIGHT_SHOULDER, 0], kp[LEFT_SHOULDER, 0]],
            [kp[RIGHT_SHOULDER, 1], kp[LEFT_SHOULDER, 1]],
            [kp[RIGHT_SHOULDER, 2], kp[LEFT_SHOULDER, 2]],
            c='g', linewidth=2, label='Shoulders')
    
    ax.plot([kp[RIGHT_ANKLE, 0], kp[LEFT_ANKLE, 0]],
            [kp[RIGHT_ANKLE, 1], kp[LEFT_ANKLE, 1]],
            [kp[RIGHT_ANKLE, 2], kp[LEFT_ANKLE, 2]],
            c='m', linewidth=2, label='Ankles')
    
    # (ส่วนคำนวณมุมในระนาบ x-z เหมือนเดิม)
    v_shoulder = np.array([kp[LEFT_SHOULDER, 0] - kp[RIGHT_SHOULDER, 0],
                           kp[LEFT_SHOULDER, 2] - kp[RIGHT_SHOULDER, 2]])
    v_ankle = np.array([kp[LEFT_ANKLE, 0] - kp[RIGHT_ANKLE, 0],
                        kp[LEFT_ANKLE, 2] - kp[RIGHT_ANKLE, 2]])
    
    dot_v = np.dot(v_shoulder, v_ankle)
    norm_v = np.linalg.norm(v_shoulder) * np.linalg.norm(v_ankle)
    angle_raw = 0 if norm_v == 0 else np.degrees(np.arccos(np.clip(dot_v / norm_v, -1.0, 1.0)))
    
    # เวกเตอร์ตั้งฉากในระนาบ x-z
    n_shoulder = np.array([-v_shoulder[1], v_shoulder[0]])
    n_ankle = np.array([-v_ankle[1], v_ankle[0]])
    
    dot_n = np.dot(n_shoulder, n_ankle)
    norm_n = np.linalg.norm(n_shoulder) * np.linalg.norm(n_ankle)
    angle_normal = 0 if norm_n == 0 else np.degrees(np.arccos(np.clip(dot_n / norm_n, -1.0, 1.0)))
    
    threshold = 45
    posture = "Standing Straight" if abs(angle_normal) < threshold or abs(180 - angle_normal) < threshold else "Twisting"
    
    ax.set_title(f"Time: {time_val:.2f}s\n"
                 f"Direct Angle: {angle_raw:.2f}°, Normal Angle: {angle_normal:.2f}°\n"
                 f"Posture: {posture}",
                 pad=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ปรับขอบเขตแกนเล็กน้อย
    padding = 0.1
    ax.set_xlim(kp[:, 0].min() - padding, kp[:, 0].max() + padding)
    ax.set_ylim(kp[:, 1].min() - padding, kp[:, 1].max() + padding)
    ax.set_zlim(kp[:, 2].min() - padding, kp[:, 2].max() + padding)
    
    ax.legend(loc='upper right')
    
    # ปรับมุมมองให้ X ซ้าย–ขวา, Y ล่าง–บน, Z เข้า–ออกจากหน้าจอ
    # ax.view_init(elev=10, azim=-90)
    ax.view_init(elev=-90, azim=-90)
    # ลองปรับ elev, azim ได้ตามต้องการ เช่น elev=20, azim=-90, ฯลฯ

    # แปลง figure เป็นภาพในรูปแบบ OpenCV (BGR)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    plt.close(fig)
    
    return img


