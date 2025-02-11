import argparse
import os
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_stream_predictor
from sam2.sam2_stream_predictor import SAM2Stream
from scipy.spatial.transform import Rotation

from pe.utils.basler import Basler, ImageLoader, Intrinsic
from pe.utils.visualizer import draw_mask, mask_center
from scripts.sam2stream import load_masks_from_source

plc_path = "D:\\Workspace\\remote_solution"
sys.path.append(plc_path)
import pymcprotocol
from plc_const import *


def euler_to_matrix(angles, order):
    
    def rotation_matrix_x(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    
    def rotation_matrix_y(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    
    def rotation_matrix_z(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    
    rotation_matrices = {
        'X': rotation_matrix_x,
        'Y': rotation_matrix_y,
        'Z': rotation_matrix_z
    }
    
    I = np.eye(3)
    
    for axis, angle in zip(order, angles):
        I = np.dot(rotation_matrices[axis](angle), I)
    
    return I

#angles = (np.pi / 4, np.pi / 6, np.pi / 3)  
#order = 'XYZ'  
#rotation_matrix = euler_to_matrix(angles, order)
#print(rotation_matrix)

def deproject_px_to_xyz(pixel, distance: float, intrinsic: Intrinsic):
    # azure distortion : k1, k2, p1, p2, k3, k4, k5, k6
    fx, fy, cx, cy = intrinsic.get_fxfycxcy()
    distortion = intrinsic.distortion
    x = (pixel[0] - cx) / fx
    y = (pixel[1] - cy) / fy

    r2 = x * x + y * y
    f = 1 + distortion[0] * r2 + distortion[1] * r2 * r2 + distortion[4] * r2 * r2 * r2
    if len(distortion) == 8:
        f += distortion[5] * (r2**4) + distortion[6] * (r2**5) + distortion[7] * (r2**6)

    ux = x * f + 2 * distortion[2] * x * y + distortion[3] * (r2 + 2 * x * x)
    uy = y * f + 2 * distortion[3] * x * y + distortion[2] * (r2 + 2 * y * y)
    x = ux
    y = uy

    z = distance
    x = x * z
    y = y * z

    return x, y, z


def draw_result(canvas, obj_info):
    height, width = canvas.shape[:2]
    fontscale = np.ceil(height / 180) / 2
    font_thick = round(fontscale)
    (txt_w, txt_h), baseline = cv2.getTextSize("obj", 1, fontscale, font_thick)
    txt_h = txt_h + baseline * 2
    from_px = obj_info[1]["center"]
    to_px = obj_info[2]["center"]
    angle = obj_info["angle"]
    translation = obj_info["translation"]
    
    

    draw_mask(canvas, obj_info[1]["mask"], color=(0, 0, 255))
    draw_mask(canvas, obj_info[2]["mask"], color=(0, 255, 0))

    # center cross lines
    cv2.line(
        canvas,
        (0, height // 2),
        (width, height // 2),
        (255, 255, 255),
        thickness=1,
    )
    cv2.line(
        canvas,
        (width // 2, 0),
        (width // 2, height),
        (255, 255, 255),
        thickness=1,
    )

    thickness = 3
    # cv2.line(canvas, from_px, (width, from_px[1]), (255, 255, 255), thickness)
    cv2.line(canvas, from_px, to_px, (0, 255, 255), thickness)

    cv2.putText(
        canvas,
        f"angle: {angle:.3f}",
        (txt_h, txt_h),
        1,
        fontscale,
        (255, 255, 255),
        font_thick * 3,
    )
    cv2.putText(
        canvas,
        f"angle: {angle:.3f}",
        (txt_h, txt_h),
        1,
        fontscale,
        (0, 0, 0),
        font_thick,
    )

    t = np.array2string(
        translation * 1000, formatter={"float_kind": lambda x: "%.1f" % x}
    )
    cv2.putText(
        canvas,
        f"t: {t}",
        (txt_h, txt_h * 2),
        1,
        fontscale,
        (255, 255, 255),
        font_thick * 3,
    )
    cv2.putText(
        canvas,
        f"t: {t}",
        (txt_h, txt_h * 2),
        1,
        fontscale,
        (0, 0, 0),
        font_thick,
    )

    return canvas


class Estimator:
    def __init__(self, config, checkpoint, device=None):
        ### select the device for computation
        if device is None:
            device = self._select_device()
        else:
            device = torch.device(device)

        print(f"using device: {device}")

        ### Loading the SAM 2 video predictor
        self.predictor: SAM2Stream = build_sam2_stream_predictor(
            config, checkpoint, device=device
        )

        self.RUN = False
        self.intrinsic = None
        self.objs = {}

    def _select_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        return device

    def set_image_info(self, roi: np.ndarray, intrinsic: Intrinsic):
        self.image_origin = np.array([roi[0], roi[1]])
        self.intrinsic = intrinsic

    def load_template(self, template_dir):
        """
        template_dir/
            L *.jpg : RGB image
            L 001/ : object_id
                L *.png : mask image
            L 002/ : object_id
                L *.png : mask image
        """
        print("Loading templates ...")
        frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(template_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]

        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # Load template color image. Covert to tensor
        self.predictor.load_template_frames(template_dir)
        # Set template mask
        template_masks = load_masks_from_source(template_dir, per_obj_png_file=True)
        for fname, template in template_masks.items():
            if fname in frame_names:
                input_frame_idx = frame_names.index(fname)
                print("input_frame_idx", input_frame_idx)
                per_obj_input_mask = template["per_obj_input_mask"]
                # input_palette = template["input_palette"]
                for object_id, object_mask in per_obj_input_mask.items():
                    self.predictor.add_new_mask(
                        frame_idx=input_frame_idx,
                        obj_id=object_id,
                        mask=object_mask,
                    )

    def _segment(self, image):
        height, width = image.shape[:2]
        out_obj_ids, out_mask_logits = self.predictor.track(image)
        if len(out_obj_ids) < 2:
            print("No object detected")
            return False

        for i, obj_id in enumerate(out_obj_ids):
            out_mask = (
                (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy()
            ).reshape(
                height, width
            )  # (w, h, batch_size)

            center = mask_center(out_mask)

            self.objs[obj_id] = {
                "mask": out_mask,
                "center": np.array(center),
            }

        return True

    def _calc_pose(self, distance=0):
        from_px = self.objs[1]["center"] + self.image_origin
        to_px = self.objs[2]["center"] + self.image_origin

        vx = np.append(to_px - from_px, [0])
        vx = vx / np.linalg.norm(vx)
        vz = np.array([0, 0, 1])
        vy = np.cross(vz, vx)

        pose = np.eye(4)
        if distance != 0 and self.intrinsic is not None:
            Rmat = np.array([vx, vy, vz]).T
            center_pt = deproject_px_to_xyz(from_px, distance, self.intrinsic)
            pose[:3, :3] = Rmat
            pose[:3, 3] = center_pt

        self.objs["pose"] = pose
        self.objs["angle"] = Rotation.from_matrix(pose[:3, :3]).as_euler(
            "zyx", degrees=True
        )[0]
        self.objs["translation"] = pose[:3, 3]

    def run(self, image, distnace=0):
        self.RUN = True
        print("Segmenting ...")
        e = datetime.now()
        ret = self._segment(image)
        print((datetime.now() - e).total_seconds())
        if ret:
            print("Calculating pose ...")
            self._calc_pose(distnace)
        self.RUN = False

    def is_running(self):
        return self.RUN

    def get_result(self):
        if "pose" in self.objs.keys():
            return True, self.objs
        else:
            return False, {}

    def reset(self):
        self.objs = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./sam/checkpoints/sam2.1_hiera_small.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--top_template_dir1",
        "-tt1",
        type=str,
        #required=True,
        default="./dataset/templates/top_btn01",
        help="directory to save the output masks (as PNG files)",
    )

    parser.add_argument(
        "--top_template_dir2",
        "-tt2",
        type=str,
        #required=True,
        default="./dataset/templates/top_btn02",
        help="directory to save the output masks (as PNG files)",
    )

    parser.add_argument(
        "--top_template_dir3",
        "-tt3",
        type=str,
        #required=True,
        default="./dataset/templates/top_btn03",
        help="directory to save the output masks (as PNG files)",
    )

    parser.add_argument(
        "--under_template_dir1",
        "-ut1",
        type=str,
        #required=True,
        default="./dataset/templates/under_btn01",
        help="directory to save the output masks (as PNG files)",
    )

    parser.add_argument(
        "--under_template_dir2",
        "-ut2",
        type=str,
        #required=True,
        default="./dataset/templates/under_btn02",
        help="directory to save the output masks (as PNG files)",
    )

    parser.add_argument(
        "--under_template_dir3",
        "-ut3",
        type=str,
        #required=True,
        default="./dataset/templates/under_btn03",
        help="directory to save the output masks (as PNG files)",
    )
    
    parser.add_argument(
        "--source",
        type=str,
        #required=True,
        help="inference input directory or camera",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default="results",
        help="save dir",
    )
    parser.add_argument(
        "--extrinsic",
        "-e",
        type=str,
        default="./calibration/under_cam/12271130/typing_Eye_to_Hand.txt",
        help="camera pose to the robot frame",
    )
    parser.add_argument(
        "--intrinsic",
        "-i",
        type=str,
        default="./calibration/camera/camera_intrinsic.json",
        help="camera pose to the robot frame",
    )
    parser.add_argument(
        "--intrinsic_under",
        "-iu",
        type=str,
        default="./calibration/camera_under/camera_intrinsic.json",
        help="camera pose to the robot frame",
    )
    parser.add_argument(
        "--plc",
        action="store_true",
        help="use plc",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ### Create Pose estimator
    top_estimator1 = Estimator(args.sam2_cfg, args.sam2_checkpoint)
    under_estimator1 = Estimator(args.sam2_cfg, args.sam2_checkpoint)
    top_estimator2 = Estimator(args.sam2_cfg, args.sam2_checkpoint)
    under_estimator2 = Estimator(args.sam2_cfg, args.sam2_checkpoint)
    top_estimator3 = Estimator(args.sam2_cfg, args.sam2_checkpoint)
    under_estimator3 = Estimator(args.sam2_cfg, args.sam2_checkpoint)
    ### Create camera controller
    #if os.path.isdir(args.source):
    #    camera = ImageLoader(args.source, ext="jpg")
    #else:
    top_camera = Basler(device_class="BaslerGigE",ip_address="192.168.100.203")
    top_camera.setConfig(exposureTime=80000.0)
    under_camera = Basler(device_class="BaslerGigE",ip_address="192.168.100.218")
    under_camera.setConfig(exposureTime=80000.0)

    if args.intrinsic:
        top_camera.load_intrinsic(args.intrinsic)
        print(top_camera.intrinsic)
    if args.intrinsic_under:
        under_camera.load_intrinsic(args.intrinsic_under)
        print(under_camera.intrinsic)

    dd = datetime.strftime(datetime.now(), "%Y-%m%d")
    save_dir = os.path.join(args.out_dir, dd)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "color"))
        os.makedirs(os.path.join(save_dir, "cropped"))
        os.makedirs(os.path.join(save_dir, "result"))

    ### load extrinsic file
   # rTc = np.loadtxt(args.extrinsic)
   # print("extrinsic\n", rTc)

    ### Load ROI info
    top_roi1 = np.loadtxt(os.path.join(args.top_template_dir1, "roi.txt")).astype(np.int32)
    rw,rh = top_roi1[2], top_roi1[3]
    cx,cy = round(top_camera.width/2),round(top_camera.height/2)
    top_roi1[0] = cx - rw//2
    top_roi1[1] = cy - rh//2 
    print(top_roi1)
    under_roi1 = np.loadtxt(os.path.join(args.under_template_dir1, "roi.txt")).astype(np.int32)
    rw,rh = under_roi1[2], under_roi1[3]
    cx,cy = round(under_camera.width/2),round(under_camera.height/2)
    under_roi1[0] = cx - rw//2
    under_roi1[1] = cy - rh//2 
    print(under_roi1)
    top_roi2 = np.loadtxt(os.path.join(args.top_template_dir2, "roi.txt")).astype(np.int32)
    rw,rh = top_roi2[2], top_roi2[3]
    cx,cy = round(top_camera.width/2),round(top_camera.height/2)
    top_roi2[0] = cx - rw//2
    top_roi2[1] = cy - rh//2 
    print(top_roi2)
    under_roi2 = np.loadtxt(os.path.join(args.under_template_dir2, "roi.txt")).astype(np.int32)
    rw,rh = under_roi2[2], under_roi2[3]
    cx,cy = round(under_camera.width/2),round(under_camera.height/2)
    under_roi2[0] = cx - rw//2
    under_roi2[1] = cy - rh//2 
    print(under_roi2)
    top_roi3 = np.loadtxt(os.path.join(args.top_template_dir3, "roi.txt")).astype(np.int32)
    rw,rh = top_roi3[2], top_roi3[3]
    cx,cy = round(top_camera.width/2),round(top_camera.height/2)
    top_roi3[0] = cx - rw//2
    top_roi3[1] = cy - rh//2 
    print(top_roi3)
    under_roi3 = np.loadtxt(os.path.join(args.under_template_dir3, "roi.txt")).astype(np.int32)
    rw,rh = under_roi3[2], under_roi3[3]
    cx,cy = round(under_camera.width/2),round(under_camera.height/2)
    under_roi3[0] = cx - rw//2
    under_roi3[1] = cy - rh//2 
    print(under_roi3)
    
    ### Load template
    # load_template(predictor, args.template_dir)
    print(args.top_template_dir1)
    top_estimator1.set_image_info(top_roi1, top_camera.intrinsic)
    top_estimator1.load_template(args.top_template_dir1)

    under_estimator1.set_image_info(under_roi1, under_camera.intrinsic)
    under_estimator1.load_template(args.under_template_dir1)

    top_estimator2.set_image_info(top_roi2, top_camera.intrinsic)
    top_estimator2.load_template(args.top_template_dir2)

    under_estimator2.set_image_info(under_roi2, under_camera.intrinsic)
    under_estimator2.load_template(args.under_template_dir2)

    top_estimator3.set_image_info(top_roi3, top_camera.intrinsic)
    top_estimator3.load_template(args.top_template_dir3)

    under_estimator3.set_image_info(under_roi3, under_camera.intrinsic)
    under_estimator3.load_template(args.under_template_dir3)

    # robot state
    if args.plc:
        pymc3e = pymcprotocol.Type3E()
        pymc3e.setaccessopt(commtype="binary")
        pymc3e.connect("192.168.1.30", 5010)

    distance_top = 1.17  # [m]
    distance_under = 0.636346306
    select_camera = 'top'
    k = ord("1")
    trigger = True
    reference_axis_deg = 0
    plc_flag_top = 0 # test
    plc_flag_under = 0 # test
    plc_case = 0 #tests
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    estimator = None
    visual_flag = 0
    flag_state = 0
    flag_run = 0
    cropped = []
    while True:
        # plc flag check
        if args.plc:
            plc_top_model = pymc3e.randomread(["D150222"],[])[0][0] # remote controller model
            plc_top_case = pymc3e.randomread(["D150212"],[])[0][0]
            plc_under_case = pymc3e.randomread(["D150220"],[])[0][0]
            plc_flag_top = pymc3e.randomread(["D150202"],[])[0][0]
            plc_flag_under = pymc3e.randomread(["D150210"],[])[0][0]
            
            print(plc_top_model, plc_top_case, plc_under_case, plc_flag_top, plc_flag_under)
            
        else:
            plc_flag = k - ord("0")

        if isinstance(top_camera, ImageLoader):
            if trigger:
                top_ret, top_image = top_camera.get()
        else:
            top_ret, top_image = top_camera.get()
        
        if isinstance(under_camera, ImageLoader):
            if trigger:
                under_ret, under_image = under_camera.get()
        else:
            under_ret, under_image = under_camera.get()

        if not top_ret or not under_ret:
            break
        stamp = datetime.strftime(datetime.now(), "%H%M%S")
        
        if plc_flag_top == 1: #and plc_flag_under == 1:
            select_camera = 'top'
            distance = distance_top
            if plc_top_case == 1:
                roi = top_roi1
                estimator = top_estimator1
                reference_axis_deg=0
            elif plc_top_case == 2:
                roi = top_roi2
                estimator = top_estimator2
                reference_axis_deg=-135
            elif plc_top_case == 3:
                roi = top_roi3
                estimator = top_estimator3
                reference_axis_deg = -90
                
            cropped = top_image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            result_img = np.zeros((roi[2], roi[3], 3), dtype=np.uint8)

        elif plc_flag_top ==0 and plc_flag_under == 1:
            select_camera = 'under'
            distance = distance_under
            if plc_under_case == 1:
                roi = under_roi1
                estimator = under_estimator1
                reference_axis_deg = 0
            elif plc_under_case == 2:
                roi = under_roi2
                estimator = under_estimator2
                reference_axis_deg = -135
            elif plc_under_case == 3:
                roi = under_roi3 
                estimator = under_estimator3
                reference_axis_deg = -135
            cropped = under_image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            result_img = np.zeros((roi[2], roi[3], 3), dtype=np.uint8)
        
        
        if (plc_flag_top == 1 or plc_flag_under == 1) and not estimator == None and flag_run == 0:
            if not estimator.is_running() and flag_state==0:
                flag_state = 1
                flag_run = 1
         
                in_img = cropped.copy()
                est_thread = threading.Thread(target=estimator.run, args=(in_img, distance))
                est_thread.start()
                result_img = cv2.putText(
                    in_img.copy(), "Processing ...", (10, 50), 1, 3, (0, 0, 255), 2
                )

            #cv2.imwrite(os.path.join(save_dir, "color", f"{stamp}.png"), image)
            #cv2.imwrite(os.path.join(save_dir, "cropped", f"{stamp}.png"), in_img)

        elif not estimator == None:
            visual_flag = 1
            if not estimator.is_running():
                
                _ret, obj_info = estimator.get_result()
    
                if _ret:
                    angles = (0, np.pi, 0)  
                    order = 'ZYX'  
                    rotation_matrix = euler_to_matrix(angles, order)      
                    pose = obj_info['pose']
                    
                    angle = 0
                    if plc_under_case == 1:
                        angle = reference_axis_deg + obj_info["angle"]
                    elif plc_under_case == 2:
                        angle = reference_axis_deg - obj_info["angle"]    
                    elif plc_under_case == 3:
                        angle = reference_axis_deg - obj_info["angle"] 
                    
                    translation = obj_info["translation"]
                    print("camera_translation",translation)
                    delta_translation=np.dot(rotation_matrix,translation)
                    obj_info["translation"] = delta_translation
                    print("angle: ", angle)
                    print("robot_translation: ", delta_translation)
                    
                    # plc write
                    if args.plc:
                        agl = int(angle * 1000)
                        dx = int(delta_translation[0] * 1000 * 1000)
                        dy = int(delta_translation[1] * 1000 * 1000)
                        print("dx: ", dx, "dy: ", dy)
                        if select_camera == 'top':
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['x']}"], [dx])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['y']}"], [dy])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['z']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rx']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['ry']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rz']}"], [agl])
                            
                        elif select_camera == 'under':
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['x1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['y1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['z1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rx1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['ry1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rz1']}"], [agl])
                    estimator.reset()
                    estimator = None
                    plc_flag_top = 0
                    plc_flag_under = 0
                    flag_state = 0
                    flag_run = 0

                    result_img = draw_result(in_img.copy(), obj_info)
                    #cv2.imwrite(
                    #    os.path.join(save_dir, "result", f"{stamp}.png"), result_img
                    #)
                else:
                    if args.plc:
                        if select_camera == 'top':
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['x']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['y']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['z']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rx']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['ry']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rz']}"], [0])
                        elif select_camera == 'under':
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['x1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['y1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['z1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rx1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['ry1']}"], [0])
                            pymc3e.randomwrite([], [], [f"D{WRITE_ADDRS[1]['rz1']}"], [0])
                    estimator.reset()
                    estimator = None
                    plc_flag_top = 0
                    plc_flag_under = 0
                    flag_state = 0
                    flag_run = 0
        
        if not estimator == None or visual_flag==1:
            # visualize
            if select_camera =='top':
                image = top_image
            elif select_camera =='under':
                image = under_image
                
            canvas = cv2.vconcat([cropped, result_img])
            h, w = image.shape[:2]
            dh = canvas.shape[0]
            dw = round(dh / h * w)
            resized_img = cv2.resize(image, (dw, dh))
            canvas = cv2.hconcat([resized_img, canvas])
            cv2.imshow("result", canvas)
            k = cv2.waitKey(1)
            if k == ord("q"):
                if estimator.is_running():
                    est_thread.join(3)
                break
            elif k < 0:
                trigger = False
            else:
                trigger = True
