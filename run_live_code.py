#Script for running new gests and trying hacks with OP for smoother predictions

import pygetwindow
import pyautogui
import PIL

import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pickle
import tkinter as tk #for gui

import datetime

import math

import os

import traceback

from termcolor import colored

import threading

from scipy.spatial import distance

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM, Dropout

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.geometry import EulerZXY
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import frame_helpers, math_helpers, robot_command, ResponseError, RpcError
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         get_se2_a_tform_b)

VELOCITY_BASE_SPEED = 0.4  # m/s
VELOCITY_BASE_ANGULAR = 0.6  # rad/sec
VELOCITY_CMD_DURATION = 0.25  # seconds
COMMAND_INPUT_RATE = 0.1

HEIGHT_MAX = 0.3  # m
HEIGHT_MAX = 0.1  # m

spot_current_body_height = 0.0

GESTURES = ["neutral", "left_arm_up", "right_arm_up", "t-pose", "left_arm_v", "both_arms_up", "squat", "right_arm_v"]

def getRobotImage():

    p = pyautogui.screenshot()
    p = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

    p = p[600:-50, 20:-30] # Crop bottom

    return p

class TakeCameraLatestPictureThread(threading.Thread):
    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        super().__init__()
        self.daemon = True
        self.start()
        
    def run(self):
        while True:
            ret, self.frame = self.camera.read()

# Live data processor
# OP gives numpy array of shape (num_people, num_joints, axis_confidence) e.g. (1, 25, 3)
def process_data(bodypoint_array, image_width, previous_points):
    processed_array = np.zeros((25, 2))
    
    # Find person in the center of the image:
        
    print("People detected: {}".format(len(bodypoint_array)))
        
    people = []
        
    for i, person in enumerate(bodypoint_array):
        neckConf = person[1][2]*100.0
        hipConf = person[8][2]*100.0
               
        # Filter out wrong detections
        if neckConf < 25 or hipConf < 25:
            print("Discarding person with neckConf {} and hipConf {}".format(neckConf, hipConf))
            continue
        
        #print(person)
        print("Person {}".format(i))
        print("  Neck: x:{0:}, y:{1:} (conf: {2:.1f}%)".format(person[1][0], person[1][1], person[1][2]*100.0))
        print("  MidHip: x:{0:}, y:{1:} (conf: {2:.1f}%)".format(person[8][0], person[8][1], person[8][2]*100.0))
        print("  Pixel distance neck->hip: {}".format(distance.euclidean([person[1][0], person[1][1]], [person[8][0], person[8][1]])))
        
        people.append({'neck': [person[1][0], person[1][1], person[1][2]*100.0],
                       'hip': [person[8][0], person[8][1], person[8][2]*100.0],
                       'pixel_position': person[1][0],
                       'distance': distance.euclidean([person[1][0], person[1][1]], [person[8][0], person[8][1]])})

    # Find middle person
    closest_position = 0
    closest_index = -1
    for i, person in enumerate(people):
        if np.abs(person['neck'][0] - image_width/2) < np.abs(closest_position - image_width/2):
            closest_position = person['neck'][0]
            closest_index = i
    
    print("Closest: {}".format(closest_index))
    
    if closest_index != -1:
        keys = bodypoint_array[closest_index]
        
        for i in range(len(keys)):
            if keys[i][2] == 0:
                keys[i] = previous_points[i]
        
            processed_array[i][0] = keys[i][0]
            processed_array[i][1] = keys[i][1]

        processed_array = np.reshape(processed_array, (1,50))
        previous_points = keys
    
    else:
        return {"pose": None, "center": None, "all": None}, previous_points
        
    return {"pose": processed_array, "center": people[closest_index], "all": people}, previous_points


# Function for calculating angle
#def calc_angle(a, b, c):
#    # b always has to be middle point
#    ba = a - b
#    bc = c - b

#    if (b==a).all() | (b==c).all():
#        return np.degrees(0.0)

#    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#    angle = np.arccos(cosine_angle)
#    return np.degrees(angle)

def calc_angle(a, b, c):

    ang= math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    
    if ang < -180:
        ang = ang + 360
    elif ang> 180:
        ang = ang - 360
    return ang

def get_6_feature_angles(X_train):

    x, y = X_train.shape
    angles = np.zeros((1, x, 6))

    for data in range(x):

        ang_right_elbow = calc_angle(np.array((X_train[data, 4], X_train[data, 5])),
                                     np.array((X_train[data, 6], X_train[data, 7])),
                                     np.array((X_train[data, 8], X_train[data, 9])))
        ang_right_shoulder = calc_angle(np.array((X_train[data, 2], X_train[data, 3])),
                                        np.array((X_train[data, 4], X_train[data, 5])),
                                        np.array((X_train[data, 6], X_train[data, 7])))
        ang_right_knee = calc_angle(np.array((X_train[data, 18], X_train[data, 19])),
                                    np.array((X_train[data, 20], X_train[data, 21])),
                                    np.array((X_train[data, 22], X_train[data, 23])))
        ang_left_elbow = calc_angle(np.array((X_train[data, 10], X_train[data, 11])),
                                    np.array((X_train[data, 12], X_train[data, 13])),
                                    np.array((X_train[data, 14], X_train[data, 15])))
        ang_left_shoulder = calc_angle(np.array((X_train[data, 2], X_train[data, 3])),
                                       np.array((X_train[data, 10], X_train[data, 11])),
                                       np.array((X_train[data, 12], X_train[data, 13])))
        ang_left_knee = calc_angle(np.array((X_train[data, 24], X_train[data, 25])),
                                   np.array((X_train[data, 26], X_train[data, 27])),
                                   np.array((X_train[data, 28], X_train[data, 29])))

        # Could have had a third for-loop for these 6 lines
        angles[0, data][0] = ang_right_shoulder
        angles[0, data][1] = ang_left_shoulder
        angles[0, data][2] = ang_right_elbow
        angles[0, data][3] = ang_left_elbow
        angles[0, data][4] = ang_right_knee
        angles[0, data][5] = ang_left_knee

    return angles

# Function for writing predictions to file, just so I can look back afterwards
def write_preds_for_frame(path, proba):
    f = open(path, 'a')
    f.write("\n\nThe HIGHEST predicted class: ")
    f.write(str(np.argmax(proba)))
    for frame in range(len(proba)):
        f.write("\nPrediction confidence for class ")
        f.write(str(frame))
        f.write(str(proba[frame]))
    f.close()
    return

def run_gui(proba):
    window = tk.Tk()
    pred = proba[0]
    txt = ''
    start_str = '\nPrediction for class '
    cl = 0
    for gest in pred:
        txt = txt+start_str+str(cl)+' is '+str(round(gest, 5))
        cl += 1

    preded_cl = '\nThe predicted class: '
    preded_cl = preded_cl+str(np.argmax(proba))

    pred_class = tk.Label(text=preded_cl, font=("Arial", 35))
    pred_class.pack()
    pred_text = tk.Label(text=txt, width=80, font=("Arial", 35))
    pred_text.pack()

    window.after(700, window.destroy) #Gui window is destroyed after 2 sec
    window.mainloop()
    return

def print_angles(angles):

    #print(angles)
    #exit(0)

    #print(angles[0])

    print("Angles:")
    #for pose in angles[0]:
    #    print('  right shoulder: {},\n  left shoulder: {},\n  right elbow: {},\n  left elbox: {},\n  right knee: {},\n  left knee: {}\n'.format(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]))
    #    break
    print('  right shoulder: {},\n  left shoulder: {},\n  right elbow: {},\n  left elbox: {},\n  right knee: {},\n  left knee: {}\n'.format(angles[0][-1][0], angles[0][-1][1], angles[0][-1][2], angles[0][-1][3], angles[0][-1][4], angles[0][-1][5]))
    
    print()

def print_classes(proba):
    print("Predicted class: {}".format(np.argmax(proba)))
    for i, probability in enumerate(proba[0]):
        print("{0}: {1:.1f}% ".format(i, probability*100), end='')
    print()
    
def get_angle_distance(pose):
    print("Getting angle and distance")

###########################
# Spot control functions: #
###########################
def _try_grpc(desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            print("Failed {}: {}".format(desc, err))
            return None


def _start_robot_command(desc, command_proto, end_time_secs=None):

        def _start_command():
            command_client.robot_command(command=command_proto,
                                         end_time_secs=end_time_secs)

        _try_grpc(desc, _start_command)
        #print("Sending robot command!")
    
def _velocity_cmd_helper(desc='', v_x=0.0, v_y=0.0, v_rot=0.0):

        print("***************************************************")
        print("{}".format(spot_current_body_height))
        print("***************************************************")
        
        params=RobotCommandBuilder.mobility_params(body_height=spot_current_body_height)

        _start_robot_command(
            desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot, params=params),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)
    
def _move_forward():
    _velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

def _move_backward():
    _velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

def _strafe_left():
    _velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

def _strafe_right():
    _velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

def _turn_left():
    _velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

def turn_right():
    _velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)
    

def relative_move(dx, dy, dyaw, robot_command_client, robot_state_client, stairs=False):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=-dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=VISION_FRAME_NAME, params=RobotCommandBuilder.mobility_params(body_height=spot_current_body_height, stair_hint=stairs))
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach the goal")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at the goal.")
            return True
        time.sleep(1)

    return True
    
def print_timers(timers):
    print("Timers:")
    for timer in timers:
        print("    {0:.5f}s ({1:.1f}hz): {2}".format(timer[0], 1/timer[0], timer[1]))
    print("\n")
    
def _orientation_cmd_helper(robot_command_client, yaw=0.0, roll=0.0, pitch=0.0, height=0.0):
        """Helper function that commands the robot with an orientation command;
        Used by the other orientation functions.

        Args:
            yaw: Yaw of the robot body. Defaults to 0.0.
            roll: Roll of the robot body. Defaults to 0.0.
            pitch: Pitch of the robot body. Defaults to 0.0.
            height: Height of the robot body from normal stand height. Defaults to 0.0.
        """

        orientation = EulerZXY(yaw, roll, pitch)
        cmd = RobotCommandBuilder.synchro_stand_command(body_height=height,
                                                        footprint_R_body=orientation)
        robot_command_client.robot_command(cmd, end_time_secs=time.time() + VELOCITY_CMD_DURATION)

def spot_rotate(angle):
    relative_move(0, 0, angle, command_client, state_client)


def spot_sit():
    robot_command.blocking_sit(command_client)
    
def spot_stand():
    robot_command.blocking_stand(command_client)

def spot_stretch():
    _orientation_cmd_helper(command_client, height=HEIGHT_MAX)
    
def spot_kneel():
    _orientation_cmd_helper(command_client, height=-HEIGHT_MAX)
    
def spot_picture():
    _orientation_cmd_helper(command_client, height=HEIGHT_MAX, pitch=-0.5)
    time.sleep(1)
    _orientation_cmd_helper(command_client, height=HEIGHT_MAX, pitch=-0.5, roll=0.1)
    time.sleep(1.0)
    _orientation_cmd_helper(command_client, height=HEIGHT_MAX, pitch=-0.5)
    time.sleep(1)
    _orientation_cmd_helper(command_client, height=HEIGHT_MAX)
    
def spot_walk_forward():
    _move_forward()
    
def spot_walk_stairs():

    _start_robot_command(
            'Walk stairs', RobotCommandBuilder.synchro_velocity_command(v_x=0.2, v_y=0, v_rot=0, params=RobotCommandBuilder.mobility_params(stair_hint=True)),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)
    
def spot_walk_backward():
    _move_backward()
    
def spot_set_pose(spot_robot_height):
    global spot_current_body_height

    if robot_height_labels[spot_robot_height] == 'sit':
        robot_command.blocking_sit(command_client)
        spot_current_body_height = 0.0
    
    elif robot_height_labels[spot_robot_height] == 'crouch':
        _orientation_cmd_helper(command_client, height=-HEIGHT_MAX)
        spot_current_body_height = -HEIGHT_MAX
        
    elif robot_height_labels[spot_robot_height] == 'normal':
        robot_command.blocking_stand(command_client)
        spot_current_body_height = 0.0
    
    elif robot_height_labels[spot_robot_height] == 'stretched':
        _orientation_cmd_helper(command_client, height=HEIGHT_MAX)
        spot_current_body_height = HEIGHT_MAX
        
    else:
        print("Unknown robot height")
        exit(-1)
        
    robot_state = robot_state_labels.index('idle')
    
if __name__ == "__main__":

    robot_standing = False
    dry_run = True
    #dry_run = False
    
    feat = 6
    buffer_len = 5
    # Load model
    #loaded_model = pickle.load(open('model_buffer5_pickle_file', 'rb'))
    #loaded_model = pickle.load(open('model_cropped_more_win8_pickle_file', 'rb'))
    #loaded_model = pickle.load(open('model_fps2_win5_g10_crop_pickle_file', 'rb'))

    ACTIONS = 8
    SEQ_LEN = 5
    JOINTS = 25
    JOINT_FEATURES = 2
    features = JOINTS*JOINT_FEATURES

    HIDDEN_UNITS = 256 # try out different number of units.
    BATCH_SIZE = 32#16 # BS=8 gives 11 samples at each batch
    EPOCHS = 200
    DROPOUT = 0.2
    LR = 1e-4 # learning rate
    OPTIMIZER = keras.optimizers.RMSprop(lr=LR)

    feat = 6
    loaded_model = Sequential()
    loaded_model.add(LSTM(HIDDEN_UNITS, input_shape=(SEQ_LEN, feat),return_sequences=True, dropout=DROPOUT))
    loaded_model.add(LSTM(HIDDEN_UNITS, dropout=DROPOUT))
    loaded_model.add(Dense(ACTIONS, activation='softmax'))
    loaded_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['acc'])

    print(type(loaded_model))

    #model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/new_model.200-0.92.h5'
    #model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/model.200-0.91_new_gests_2_win8.h5'
    #model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/model.200-0.93_win8_g5synth.h5'
    #model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/model_train/models_6_gests/model.200-0.95_g3g4g7g8g9g10_win8.h5'
    #model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/model_train/new_gests_g3g4g7g8g9g10ex_win8/model.200-0.94.h5'
    #model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/model_train/new_gests_g3g4g7g8g9g10ex_win5/model.200-0.95.h5'
    model_file = 'C:/Users/ffi-win/Desktop/openpose/build/examples/tutorial_api_python/model_g08Right.200-0.92.h5'
    loaded_model.load_weights(model_file)
    
    # Initiate robot:
    bosdyn.client.util.setup_logging()

    sdk = bosdyn.client.create_standard_sdk('testSitStand')

    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    ignore_timer = int(time.time()) + 8 # Don't do an action for the first 15 seconds

    robot_height_labels = ['sit', 'crouch', 'normal', 'stretched']
    robot_height = 0 # state variable with above labels
    robot_state_labels = ['idle', 'ready_to_walk_to_person', 'walking_to_person']
    robot_state = 0 # state variable with above labels
    robot_stair_mode = False

    # Acquire lease
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        state_client = robot.ensure_client(RobotStateClient.default_service_name)
        state = state_client.get_robot_state()
    
        # Power On
        robot.power_on()
        assert robot.is_powered_on(), "Robot power on failed."
    
        print("Ready")

        # Running OP to get keypoints
        try:
            # Import Openpose (Windows/Ubuntu/OSX)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            try:
                # Windows Import
                if platform == "win32":
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    sys.path.append(dir_path + '/../../python/openpose/Release');
                    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                    import pyopenpose as op
                else:
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    sys.path.append('../../python');
                    # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                    # sys.path.append('/usr/local/python')
                    from openpose import pyopenpose as op

            except ImportError as e:
                print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
                raise e

            # Flags
            parser = argparse.ArgumentParser()
            parser.add_argument("--camera", default=0, help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
            parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
            args = parser.parse_known_args()

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            params = dict()
            params["model_folder"] = "../../../models/"

            # Add others in path?
            for i in range(0, len(args[1])):
                curr_item = args[1][i]
                if i != len(args[1])-1: next_item = args[1][i+1]
                else: next_item = "1"
                if "--" in curr_item and "--" in next_item:
                    key = curr_item.replace('-','')
                    if key not in params:  params[key] = "1"
                elif "--" in curr_item and "--" not in next_item:
                    key = curr_item.replace('-','')
                    if key not in params: params[key] = next_item

            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            
            datum = op.Datum()
            
            #my = pygetwindow.getWindowsWithTitle('Mozilla Firefox')[0]
            #my.activate()
            
            vid = cv2.VideoCapture(2)

            vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print("Width: {}, height: {}".format(width, height))
            
            latest_picture = TakeCameraLatestPictureThread(vid)
            
            for i in range(5):
                ret, frame = vid.read() # Dump the first few frames

            # But the model is trained on a sequnce of 83 frames, so this is what i have to give
            # sequence_size = 83 #3 #48
            buffer = np.zeros((buffer_len, 50))
            num_frames = 0
            
            points = np.zeros((25, 3))
            
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            counter = 0
            directory = "C:/Users/ffi-win/Desktop/images/{}/".format(current_time)
            
            os.mkdir(directory)
            
            pred_3_in_buffer = np.zeros((5))

            while True:
                
                start_time = time.perf_counter()
                timers = []
                
                ret, frame = vid.read()
                frame = latest_picture.frame
                frame = frame[90:330, :] # Crop bottom
                
                #frame = getRobotImage()
                
                #frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75)
                
                image_width = frame.shape[1]
                
                print(frame.shape)
                
                datum.cvInputData = frame
                
                cv2.imwrite("{}/{}.jpg".format(directory, counter), frame)
                counter += 1
                
                current_time = time.perf_counter()
                timers.append([current_time - start_time, "before OpenPose"])
                
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                
                cv2.imshow("skeleton", datum.cvOutputData)
                time.sleep(0.1)
                #cv2.waitKey(0)
                
                body_keypoints = datum.poseKeypoints

                timers.append([time.perf_counter() - current_time, "OpenPose"])
                current_time = time.perf_counter()

                if not type(body_keypoints) == np.ndarray:
                    # No person found by openpose
                    print('No people found')
                
                else:
                    # Person found by openpose
                    
                    processed_data, points = process_data(body_keypoints, image_width, points) #gives shape (1, 50)
                    
                    if processed_data["pose"] is None:
                    
                        # Person found by OP, but rejected
                        print('Person found by OP, but it was rejected')
                        
                        continue
                    
                    data = processed_data["pose"] # Pose of most center person
                    
                    pixel_position = processed_data['center']['pixel_position']
                    person_distance = processed_data['center']['distance']
                    
                    print("pixel position: {}, distance: {}".format(pixel_position, person_distance))
                    angle = ((((pixel_position / image_width) * 2 * np.pi) - np.pi)*0.95)
                    
                    # buffer data
                    if buffer[-1].any() == 0: # buffer is not filled yet
                        buffer[num_frames] = data
                    else:
                        buffer[:-1] = buffer[1:]
                        buffer[-1] = data
                    
                    num_frames += 1
                    print("{}".format(num_frames), end=' ', flush=True)
                    if (num_frames) >= buffer_len:
                        print('\n')

                        # Calculating angles
                        angles = get_6_feature_angles(buffer)
                        
                        #print(buffer[-1])
                        
                        #print_angles(angles)
                        
                        #buffer = np.zeros((buffer_len, 50))

                        timers.append([time.perf_counter() - current_time, "OpenPose -> predict"])
                        current_time = time.perf_counter()
                       
                        proba = loaded_model.predict_proba(angles)

                        timers.append([time.perf_counter() - current_time, "predict"])
                        current_time = time.perf_counter()
                        
                        print_classes(proba)
                        
                        pre = np.argmax(proba)
                        
                        print(proba)
                        
                        #if proba[0][1] > 0.6: #if g1 stop is greater than 60% then choose g1
                        #    pre = 1
                        #elif proba[0][2] > 0.7:
                        #    pre = 2
                        #elif proba[0][5] > 0.69:#0.75:
                        #    pre = 5
                        #elif proba[0][6] > 0.69:
                        #    pre = 6
                        if proba[0][4] > 0.79:
                            pre = 4
                        elif (pre==6) & (proba[0][pre]<0.98): #if predicts 10 and model conf is <97% choose next highest
                            pre = np.argmax(proba[0][:-1])
                            #if (proba[0][pre]<0.95):
                            #    pre = 0
                            #if pre != 2:
                            #    pre = 0
                        # Overwrite to neutral if conf less than 95%
                        elif proba[0][pre] < 0.5:
                            pre = 0
                        
                        pred_3_in_buffer[:-1] = pred_3_in_buffer[1:]
                        pred_3_in_buffer[-1] = pre
                        print("3 in buffer: ",pred_3_in_buffer)
                        u, cnt = np.unique(pred_3_in_buffer, return_counts=True)
                        three_of_same = u[cnt > 2]
                        print("Three of same within 5 predictions:",three_of_same)
                        
                        # Do extra check on T-pose to stop extra walking
                        
                        pre = int(pre)
                        
                        if GESTURES[pre] == 't-pose':
                            print("Replacing T-pose to be extra strict")
                            three_of_same = u[cnt > 3]
                        
                        if len(three_of_same) == 0: #If no three same predictions then set pre to neutral class
                            print("2")
                            pre = 0
                        else:
                            print("3")
                            pre = three_of_same[0]
                    
                        pre = int(pre)
                        
                        print('\n********\n  {}\n********\\n'.format(GESTURES[pre]))
                        
                        if dry_run:
                            print("Not doing action as dry_run is true!")
                            
                        elif int(time.time()) < ignore_timer:
                            if GESTURES[pre] == 'squat': # Stop 
                                print("We should stop now")
                        
                            print("Waiting for ignore_timer")
                        else:
                            if GESTURES[pre] == 'neutral':
                                robot_state = robot_state_labels.index('idle')
                        
                            if GESTURES[pre] == 'squat': # Stop 
                                print("We should stop now")
                            
                            elif GESTURES[pre] == 'both_arms_up': # Stair mode
                                
                                if robot_height == robot_height_labels.index('normal'):
                                    spot_walk_stairs()
                                
                            elif GESTURES[pre] == 'left_arm_v': # Picture

                                if robot_height_labels[robot_height] == 'stretched':
                                    spot_picture()
                                    ignore_timer = int(time.time()) + 5
                                
                            elif GESTURES[pre] == 't-pose': # Walk
                                if robot_height == 0:
                                    print("Not walking since the robot is on the ground!")
                                else:
                                    spot_walk_forward()
                                
                            elif GESTURES[pre] == 'left_arm_up': # Sit / crouch
                                if robot_height > 0:
                                    robot_height -= 1
                                
                                spot_set_pose(robot_height)
                                
                                ignore_timer = int(time.time()) + 3

                            elif GESTURES[pre] == 'right_arm_up': # stand/stretch
                                if robot_height_labels[robot_height] == 'sit':
                                    robot_height = robot_height_labels.index('normal')
                                elif robot_height < len(robot_height_labels)-1:
                                    robot_height += 1
                                    
                                spot_set_pose(robot_height)
                                
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                print("{}".format(spot_current_body_height))
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                
                                ignore_timer = int(time.time()) + 3
                                
                            elif GESTURES[pre] == 'right_arm_v': # Look at me
                                spot_rotate(angle)
                                
                                ignore_timer = int(time.time()) + 3
                                
                            else:
                                print('No action initiated')

                            print()
                    
                    # Stop if we reached the person while walking to them
                    #if robot_state == robot_state_labels.index('walking_to_person'):
                        # If a person is found, but not in the middle:
                    #    if angle > 1.0 or angle < 1.0:
                    #        spot_set_pose(robot_height)
                    #    else:
                    #        spot_walk_forward()
                    
                    
                    
                    timers.append([time.perf_counter() - current_time, "predict -> done"])
                    current_time = time.perf_counter()
                    
                    timers.append([time.perf_counter() - start_time, "Whole loop"])
                    
                    print_timers(timers)

        except Exception as e:
            print(e)
            sys.exit(-1)
