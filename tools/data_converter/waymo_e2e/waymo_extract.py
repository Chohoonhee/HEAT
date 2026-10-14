import tensorflow as tf
import os
import json
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from PIL import Image
import numpy as np
from google.protobuf.json_format import MessageToDict

# TFRecord 파일 경로 설정
DATASET_FOLDER = './waymo_e2e_tfrecord'
TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training_*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'val_*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test_*')

# 모든 TFRecord 파일 읽기
filenames = tf.io.matching_files(VALIDATION_FILES)
dataset = tf.data.TFRecordDataset(filenames, compression_type='')
dataset_iter = dataset.as_numpy_iterator()


# 카메라 이름 정의
camera_names = ["front_left", "front", "front_right", "side_left", "side_right", "rear_left", "rear", "rear_right"]

# 디렉토리 생성 함수
def create_directories(scene_id):
    scene_dir = f'./waymo_e2e_val/{scene_id}'
    os.makedirs(scene_dir, exist_ok=True)
    for camera_name in camera_names:
        camera_dir = os.path.join(scene_dir, camera_name)
        os.makedirs(camera_dir, exist_ok=True)
    return scene_dir

# TFRecord에서 데이터 추출
def process_record(data):
    # 시퀀스 이름 추출
    sequence_name = data.frame.context.name  # UUID-SeqNum 형식으로 추출
    scene_id, seq_num = sequence_name.split('-')  # UUID-SeqNum을 분리
    
    # 디렉토리 생성
    scene_dir = create_directories(scene_id)
    
    # timestmap 출력
    timestamp = data.frame.timestamp_micros 
    # print(timestamp) # timestamp가 다 0임

    # 카메라 이미지 추출
    camera_images = return_cameras(data)
    
    # data_dict = MessageToDict(data)
    # print(data_dict['frame']['images'][0]['pose'])
    
    # 차량 상태 및 운전 의도 추출
    future_states = data.future_states
    
    

    past_states = data.past_states
    


    intent = data.intent
    preference_trajectories = data.preference_trajectories
    
    # import pdb; pdb.set_trace()
    # 카메라 캘리브레이션 정보 추출
    camera_calibrations = extract_camera_calibrations(data)
    
    velocity = extract_camera_velocity(data)
    
    
    
    # 어노테이션 저장
    save_annotation(scene_dir, seq_num, future_states, past_states, intent, preference_trajectories, \
        camera_calibrations, timestamp, velocity)
    
    # # 카메라 이미지 저장
    # for camera_name, images in camera_images.items():
    #     for i, img in enumerate(images):
    #         image_path = os.path.join(scene_dir, camera_name, f'{seq_num}.jpg')
    #         Image.fromarray(img).save(image_path)


# 카메라 이미지 추출 함수
def return_cameras(data: wod_e2ed_pb2.E2EDFrame):
    """모든 카메라 이미지 추출"""
    camera_images = {}
    for camera_name in camera_names:
        camera_images[camera_name] = []
        for image_content in data.frame.images:
            if image_content.name == getattr(open_dataset.CameraName, camera_name.upper(), None):
                image = tf.io.decode_image(image_content.image).numpy()
                camera_images[camera_name].append(image)
    return camera_images


def extract_camera_calibrations(data):
    calibrations = {}
    for camera in data.frame.context.camera_calibrations:
        camera_name = open_dataset.CameraName.Name.Name(camera.name).lower()  # enum 값을 문자열로 변환
        calibrations[camera_name] = {
            "intrinsic": list(camera.intrinsic),  # RepeatedScalarContainer를 리스트로 변환
            "extrinsic": list(camera.extrinsic.transform) if camera.extrinsic else None,  # RepeatedScalarContainer를 리스트로 변환
        }
    return calibrations

def extract_camera_velocity(data):
    velocity = {}
    for i, image_content in enumerate((data.frame.images)):
        camera_name = camera_names[i]
        vel_cam = MessageToDict(image_content.velocity)
        velocity[camera_name] = vel_cam
        
    return velocity




# 어노테이션 저장 함수
def save_annotation(scene_dir, seq_num, future_states, past_states, intent, preference_trajectories, camera_calibrations, timestamp, velocity):
    preference_lists = []
    if len(preference_trajectories) > 0 and \
      preference_trajectories[0].preference_score != -1:
      
      for trajectory in preference_trajectories:
        trajectory_data = {
            "trajectory": {
                "x": list(trajectory.pos_x),  # 리스트로 변환
                "y": list(trajectory.pos_y),  # 리스트로 변환
            },
            "score": trajectory.preference_score # -1이면 None (null)
        }
        preference_lists.append(trajectory_data)
    print(scene_dir.split('/')[-1], seq_num)
    annotation = {
        "sequence_id": scene_dir.split('/')[-1],
        "frame_id": seq_num,  # frame_id는 SeqNum만 사용
        "metadata": {
            "timestamp_micros": timestamp,  # timestamp가 없으므로 예시 값 사용
            "future_states": {
                "pos_x": list(future_states.pos_x),
                "pos_y": list(future_states.pos_y),
                "pos_z": list(future_states.pos_z) #only used for viz
            },
            "past_states": {
                "pos_x": list(past_states.pos_x),
                "pos_y": list(past_states.pos_y),
                "vel_x": list(past_states.vel_x),
                "vel_y":  list(past_states.vel_y),
                "accel_x": list(past_states.accel_x),
                "accel_y": list(past_states.accel_y)
            },
            "intent": intent,
            "preference_trajectories": preference_lists,
            "camera_calibrations": camera_calibrations,
            "velocity": velocity
        }
    }
    
    annotation_path = os.path.join(scene_dir,f'{seq_num}_annotation.json')
    with open(annotation_path, 'w') as ann_file:
        json.dump(annotation, ann_file, indent=4)

# TFRecord 파일에 있는 모든 시퀀스 처리
def process_all_sequences():
    for bytes_example in dataset_iter:
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(bytes_example)
        process_record(data)
        # break

# 데이터 처리 시작
process_all_sequences()

print("Images and annotations for all sequences have been saved successfully.")
