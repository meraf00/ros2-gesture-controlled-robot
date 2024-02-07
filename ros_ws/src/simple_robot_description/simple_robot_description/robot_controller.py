import rclpy

from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist


from .gesture_detection import *

class RobotControllerNode(Node):
    
    def __init__(self):        
        super().__init__('robot_controller')
        self.get_logger().info('Robot controller started')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)                        
        
    def send_velocity_cmd(self):
        msg = Twist()

        msg.linear.x = -30.0                        

        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = RobotControllerNode()        

    use_brect = True

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()

    path = os.path.dirname(os.path.abspath(__file__))

    # Read labels ###########################################################
    with open(
        f"{path}/gesture_model/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    #  ########################################################################
    mode = 0

    if os.getenv('environment') == 'wsl':
        client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        host_ip = '172.21.64.1'
        port = 9999
        client_socket.connect((host_ip,port))
        data = b""
        payload_size = struct.calcsize("Q")

        while True:
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024) # 4K
                if not packet: break
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q",packed_msg_size)[0]
            
            while len(data) < msg_size:
                data += client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data  = data[msg_size:]
            image = pickle.loads(frame_data)
            

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture #####################################################                        
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                    )

                    if (keypoint_classifier_labels[hand_sign_id] == 'Forward'):
                        node.send_velocity_cmd()

            debug_image = draw_info(debug_image, mode, number)

            # Screen reflection #############################################################
            cv.imshow("Hand Gesture Recognition", debug_image)
        

    else:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                    )

            debug_image = draw_info(debug_image, mode, number)

            # Screen reflection #############################################################
            cv.imshow("Hand Gesture Recognition", debug_image)

        cap.release()
        cv.destroyAllWindows()


    

    rclpy.shutdown()


if __name__ == '__main__':
    main()
