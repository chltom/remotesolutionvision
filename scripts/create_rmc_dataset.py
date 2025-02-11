import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
from pe.utils.basler import Basler
# from bgyoo_package.camera import Basler


def draw_rotated_line(image, angle=0, color=(255, 0, 0)):
    """
    ang: degree
    """

    h, w, _ = image.shape
    cx, cy = np.round(w / 2).astype(np.int32), np.round(h / 2).astype(np.int32)
    l = min(h, w) // 2

    rad = np.deg2rad(angle)
    x1 = np.round(cx + l * np.cos(rad)).astype(np.int32)
    y1 = np.round(cy + l * np.sin(rad)).astype(np.int32)
    cv2.line(image, (cx, cy), (x1, y1), color=color, thickness=3)


def main(args):
    save_dir = os.path.join(args.save_dir, f"btn_{args.btn:02}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Create {save_dir}")

    camera = Basler(**vars(args))
    intrinsic = camera.get_intrinsic()
    w, h = camera.image_shape()
    print(f"image shape : {w}x{h}")
    print(intrinsic)

    font_face = 1
    font_scale = round(h / 360)
    font_thick = font_scale + round(font_scale / 2)
    (_, font_h), baseline = cv2.getTextSize("Agy", font_face, font_scale, font_thick)
    font_h += baseline

    WINDOW_NAME = "Canvas"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    idx = 0
    fname = ""
    draw_ang_list = list(range(0, 360, 45))
    capture_ang_list = list(range(0, 360, 5))
    while idx < len(capture_ang_list):
        rgb, depth, points, ret = camera.get_image()
        canvas = rgb.copy()
        for a in draw_ang_list:
            draw_rotated_line(canvas, a)

        ang = capture_ang_list[idx]
        draw_rotated_line(canvas, ang, color=(0, 0, 255))

        canvas = cv2.putText(
            canvas,
            f"{idx:>5} angle : {ang}",
            (10, font_h * 2),
            font_face,
            font_scale,
            (0, 0, 255),
            font_thick,
        )

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1)
        fname = os.path.join(save_dir, f"{args.btn:02}_{ang:03}.png")
        if key == ord("c"):
            cv2.imwrite(fname, rgb)
            print("save ", fname)
            idx += 1
        elif key == ord("q"):
            break
    camera.stop()


def crop_images(args):
    from_dir = args.source
    save_dir = f"{from_dir}_cropped"

    print("from", from_dir)
    print("to", save_dir)

    frame_names = [
        p
        for p in os.listdir(from_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]

    if len(frame_names) > 0 and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    frame_names.sort()

    image = cv2.imread(os.path.join(from_dir, frame_names[0]))
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("select ROI", image)
    print(roi)
    dw = max(roi[2], roi[3])
    dw = int(round(dw / 2, -1))
    print(dw)
    cv2.destroyAllWindows()
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    # dw = 100
    for frame_name in tqdm(frame_names):
        image = cv2.imread(os.path.join(from_dir, frame_name))
        crooed_image = image[cy - dw : cy + dw, cx - dw : cx + dw]
        cv2.imwrite(
            os.path.join(save_dir, frame_name).replace("png", "jpg"), crooed_image
        )
        # print(crooed_image.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera test")
    parser.add_argument(
        "--camera_mode", type=int, default=1, help="0:streaming, 1:trigger"
    )
    parser.add_argument("--source", "-s", type=str, default="")
    parser.add_argument("--loop", action="store_false")
    parser.add_argument("--save_dir", type=str, default="./dataset")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--btn", type=int, default=0)
    parser.add_argument("--calibration_file", type=str, default="")

    # for basler
    gain_type = ["off", "once", "continuous"]
    parser.add_argument("--auto_gain", choices=gain_type, default="off")
    parser.add_argument("--gain_raw", type=int, default=14)
    parser.add_argument("--auto_exposure", choices=gain_type, default="off")
    parser.add_argument("--exposure_time", type=float, default=61180.0)

    args = parser.parse_args()
    args.auto_gain = gain_type.index(args.auto_gain)
    args.auto_exposure = gain_type.index(args.auto_exposure)
    # basler_config = {"gain": 0.0, "exposureTime": 7449.0}
    print(args)
    # main(args)
    crop_images(args)
