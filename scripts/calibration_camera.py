import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm
from pe.utils.basler import Basler
from pe.utils.camera_utils import ChessBoard, Intrinsic, vector2Mat, rvect2Rmat
from scipy.spatial.transform import Rotation


def intrinsicCal(board_data, out_dir):
    print("Intrinsic calibration")
    chesspts = board_data["chesspts"]
    imgpoints = board_data["imgpoints"]
    h = board_data["height"]
    w = board_data["width"]

    ret = -1
    print("Starg calculating ... ")
    if len(chesspts) > 4:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            chesspts, imgpoints, (w, h), None, None
        )
        print("Intrinsic calibration Done ", ret)
        print(mtx)
        print(dist)
        cam_param = Intrinsic(width=w, height=h, camera_matrix=mtx, distortion=dist)
        fname = os.path.join(out_dir, "camera_intrinsic.json")
        cam_param.save(fname)
        print("Save ", fname)

        Rmat_list = []
        tvec_list = []
        for rvec, tvec in zip(rvecs, tvecs):
            Rmat = rvect2Rmat(rvec.squeeze())
            Rmat_list.append(Rmat)  # origin: cam
            tvec_list.append(tvec.squeeze())
        board_data["Rmats"] = Rmat_list
        board_data["tvecs"] = tvec_list
        cv2.destroyAllWindows()

        mean_error = 0
        for i in range(len(chesspts)):
            imgpoints2, _ = cv2.projectPoints(
                chesspts[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("total error: {} [px]".format(mean_error / len(chesspts)))

    else:
        print(len(chesspts))
        print("Can't Calibration. Insufficient Data.")
    return ret, board_data


def capture(camera: Basler, save_dir, board: ChessBoard, robot=None):
    if robot is None:
        Flag_robot = False
    else:
        Flag_robot = True
    robot_poses = []

    capture_idx = 0
    info_str = "Press 'c' to capture images. If done press 'q'"
    print(info_str)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    while True:
        image = camera.get()  # bgr
        if image is None:
            print("Check camera connection!")
            break
   
        canvas = image[1].copy()

        # Detect chessboard
        retval = board.detect(canvas, scale=5)
        if retval:
            board.draw(canvas)

        cv2.putText(canvas, info_str, (20, 100), 1, 10, (0, 0, 255), thickness=3)
        cv2.putText(
            canvas,
            f"captured: {capture_idx}",
            (20, 300),
            1,
            10,
            (0, 0, 255),
            thickness=3,
        )

        cv2.imshow("image", canvas)
        k = cv2.waitKey(0)
        if k == ord("c"):  # capture
            # captured_images.append(cv2.resize(canvas, (canvas.shape[1]//4, canvas.shape[0]//4)))
            if retval:
                print(f"\nCapture {capture_idx+1} image")
                board.capture()

                if Flag_robot:
                    # robot_pose = robot.get_pose().as_xyzrot()
                    txt_pose = input("x,y,z,rx,ry,rz: ")
                    robot_pose = list(map(float, txt_pose.split(",")))
                    print("robot pose : ", robot_pose)
                    robot_poses.append(robot_pose)

                cv2.imwrite(os.path.join(save_dir, f"{capture_idx:02}.png"), image[1])
                if Flag_robot:
                    np.savetxt(
                        os.path.join(os.path.join(save_dir, "..", "robot_xyzrot.txt")),
                        robot_poses,
                    )
                capture_idx += 1
                print("Saved ", capture_idx)

            else:
                print("Can't Capture")

        elif k == ord("q"):
            break

    h, w, _ = image[1].shape
    board_data = board.get_captured_data()
    board_data["width"] = w
    board_data["height"] = h
    return board_data, robot_poses


def load_board_data(source_dir, board: ChessBoard):
    img_list = sorted(glob(os.path.join(source_dir, "*.png")))
    for img_path in tqdm(img_list):
        image = cv2.imread(img_path)
        canvas = image.copy()

        # Detect chessboard
        retval = board.detect(canvas, scale=5)
        if retval:
            board.capture()
        else:
            print("[!] Can not find board. Something Wrong.")

    h, w, _ = image.shape
    board_data = board.get_captured_data()
    board_data["width"] = w
    board_data["height"] = h

    return board_data


def load_robot_data(source_file):
    robot_poses = np.loadtxt(source_file)
    return robot_poses


def extrinsicCal(board_data, robot_poses, mode="to"):
    """
    robot_poses: list of [x,y,z,rx,ry,rz]
    mode : to (Hand-to-Eye) or in (Hand-in-Eye)
    """
    print(f"Extrinsic calibration.  Eye_{mode}_Hand")
    cam_R = board_data["Rmats"]  # camera frame, board pose
    cam_t = board_data["tvecs"]

    robot_R = []
    robot_t = []
    _robot_R = []
    _robot_t = []
    for robot_pose in robot_poses:
        # robot pose : xyz [mm], rxryrz [deg]
        pose = np.eye(4)
        pose[:3, 3] = robot_pose[:3] * 0.001
        pose[:3, :3] = Rotation.from_euler(
            "xyz", robot_pose[-3:], degrees=True
        ).as_matrix()
        if mode == "to":  # Hand to Eye
            pose = np.linalg.inv(pose)  # gTb 가 입력.

        robot_R.append(pose[:3, :3])
        robot_t.append(pose[:3, 3])

        _pose = np.linalg.inv(pose)  # gTb 가 입력.

        _robot_R.append(_pose[:3, :3])
        _robot_t.append(_pose[:3, 3])

    # robot frame camera pose
    print(len(robot_R), len(cam_t))
    R_cam2robot, t_cam2robot = cv2.calibrateHandEye(
        robot_R, robot_t, cam_R, cam_t, method=cv2.CALIB_HAND_EYE_TSAI
    )
    # board frame robot pose, camera frame robot pose
    boardRrobot, board_t_robot, camRrobot, cam_t_robot = cv2.calibrateRobotWorldHandEye(
        cam_R, cam_t, _robot_R, _robot_t
    )
    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = R_cam2robot
    extrinsic_mat[:3, 3] = t_cam2robot.squeeze()

    print(extrinsic_mat)
    _mat = np.eye(4)
    _mat[:3, :3] = camRrobot
    _mat[:3, 3] = cam_t_robot.squeeze()
    print("robot frame camera pose\n", np.linalg.inv(_mat))
    _mat = np.eye(4)
    _mat[:3, :3] = boardRrobot
    _mat[:3, 3] = board_t_robot.squeeze()
    _mat = np.linalg.inv(_mat)
    print("-- robot frame board pose\n", _mat)

    return extrinsic_mat, _mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, help="path to source directory")
    parser.add_argument("--out_dir", "-o", type=str, help="path to out directory")
    parser.add_argument("--robot", "-r", type=str, help="robot name")
    # parser.add_argument(
    #     "--robot", "-r", action="store_true", help="get robot data directly"
    # )
    parser.add_argument(
        "--intrinsic", "-i", action="store_true", help="Do intrinsic calibration"
    )
    parser.add_argument(
        "--extrinsic", "-e", type=str, help="Do extrinsic calibration, [to, in]"
    )
    args = parser.parse_args()

    stage_info = ""

    chess_l = 0.01
    chess_size = (10, 7)
    print(f"Use Chessboard. length={chess_l}, size={chess_size}")
    board = ChessBoard(length=chess_l, size=chess_size)

    if not args.source:
        Flag_capture = True
        out_dir = os.path.join(args.out_dir, f"{datetime.now().strftime('%m%d%H%M')}")
        board_dir = os.path.join(out_dir, "board")
        print("Save results to : ", out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            os.mkdir(board_dir)
    else:
        Flag_capture = False
        board_dir = os.path.join(args.source, "board")
        if not os.path.exists(board_dir):
            print(f"[!] Can not find {board_dir}")
            exit()

        out_dir = args.source

    with open(os.path.join(out_dir, "stage_info.txt"), "w") as text_file:
        text_file.write(stage_info)

    if args.robot:
        robot = args.robot
    else:
        robot = None

    pose_file = os.path.join(os.path.join(out_dir, "robot_xyzrot.txt"))
    if Flag_capture:
        print("Create Camera")
        camera_model = None
        camera = Basler(model=camera_model)
        board_data, robot_poses = capture(camera, board_dir, board, robot)
        np.savetxt(pose_file, robot_poses)
    else:
        board_data = load_board_data(board_dir, board)

    if args.intrinsic:
        rms_val, board_data = intrinsicCal(board_data, out_dir)
        print("rms ", rms_val)

    if args.extrinsic:
        if os.path.exists(pose_file):
            robot_poses = load_robot_data(pose_file)
        elif robot is None:
            print("[!] Can not find robot info")
            exit()

        extrinsic_mat, _board = extrinsicCal(board_data, robot_poses, args.extrinsic)
        np.savetxt(
            os.path.join(out_dir, f"{args.robot}_Eye_{args.extrinsic}_Hand.txt"),
            extrinsic_mat,
        )
        np.savetxt(os.path.join(out_dir, f"robot_board.txt"), _board)
