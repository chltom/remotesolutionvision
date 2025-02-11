import argparse
import os
from typing import Union

import cv2
import numpy as np
import torch

# from bgyoo_package.camera import Basler
from sam2.build_sam import build_sam2_stream_predictor
from sam2.sam2_stream_predictor import SAM2Stream
from sam2.sam2_video_predictor import SAM2VideoPredictor

from pe.utils.basler import Basler, ImageLoader
from pe.utils.cv_interface import cvInterface
from pe.utils.visualizer import *
from sam.tools.vos_inference import (
    get_per_obj_mask,
    load_ann_png,
    put_per_obj_mask,
    save_ann_png,
)


def load_masks_from_source(
    input_mask_dir, per_obj_png_file=False, allow_missing=True
) -> dict:
    """Load masks from a directory as a dict of per-object masks.
    and return with frame name
    """
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(input_mask_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort()

    masks = {}
    for frame_name in frame_names:
        if not per_obj_png_file:
            input_mask_path = os.path.join(input_mask_dir, f"{frame_name}.png")
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask = get_per_obj_mask(input_mask)

        else:
            per_obj_input_mask = {}
            input_palette = None
            for object_name in os.listdir(input_mask_dir):
                obj_dir = os.path.join(input_mask_dir, object_name)
                if not os.path.isdir(obj_dir):
                    continue

                object_id = int(object_name)
                input_mask_path = os.path.join(
                    input_mask_dir, object_name, f"{frame_name}.png"
                )
                if allow_missing and not os.path.exists(input_mask_path):
                    continue
                input_mask, input_palette = load_ann_png(input_mask_path)
                per_obj_input_mask[object_id] = input_mask > 0

        masks[frame_name] = {
            "per_obj_input_mask": per_obj_input_mask,
            "input_palette": input_palette,
        }

    return masks


def save_masks_to_dir(
    output_mask_dir,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(output_mask_dir, exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(output_mask_dir, frame_name)
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(output_mask_dir, object_name, frame_name)
            save_ann_png(output_mask_path, output_mask, output_palette)


def calc_px_angel(from_px, to_px):
    if (to_px[0] - from_px[0]) == 0:
        angle = 90
        if to_px[1] < from_px[1]:
            angle = 270
    else:
        theta = np.arctan((to_px[1] - from_px[1]) / (to_px[0] - from_px[0]))
        angle = np.rad2deg(theta)
        if angle > 0:
            if to_px[1] < from_px[1]:
                angle = angle + 180
        else:
            if to_px[1] < from_px[1]:
                angle = angle + 360
            else:
                angle = angle + 180
    return angle


def test(predictor: SAM2Stream, source_dir, save_dir, roi=None):
    frame_names = [
        p
        for p in os.listdir(source_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    height = predictor.condition_state["video_height"]
    width = predictor.condition_state["video_width"]

    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    

    while True:
        ret, image = camera.get()
        # image = cv2.imread(os.path.join(source_dir, fname))

        
        if roi is not None:
            image = image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]

        e = datetime.now()
        print("frame_idx: ", predictor.frame_idx)
        out_obj_ids, out_mask_logits = predictor.track(image)
        dt = (datetime.now() - e).total_seconds()
        print(f"{dt:.3f} [s]")

        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > 0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .reshape(height, width)
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        draw_mask(image, per_obj_output_mask[1], color=(0, 0, 255))
        draw_mask(image, per_obj_output_mask[2], color=(0, 255, 0))

        # calculate angle and visualize
        from_px = mask_center(per_obj_output_mask[1])
        to_px = mask_center(per_obj_output_mask[2])
        angle = calc_px_angel(from_px, to_px)

        fontscale = np.ceil(height / 180) / 2
        thickness = max(round(fontscale), 1)
        (txt_w, txt_h), baseline = cv2.getTextSize("ijkpq", 1, fontscale, thickness)
        cv2.line(image, from_px, (width, from_px[1]), (255, 255, 255), thickness)
        cv2.line(image, from_px, to_px, (0, 255, 255), thickness)
        cv2.putText(
            image,
            f"angle: {angle:.3f}",
            (txt_h, txt_h + baseline),
            1,
            fontscale,
            (255, 255, 255),
            thickness * 3,
        )
        cv2.putText(
            image,
            f"angle: {angle:.3f}",
            (txt_h, txt_h + baseline),
            1,
            fontscale,
            (0, 0, 0),
            thickness,
        )

        _fname = f"{fname.split('.')[0]}.png"
        if width > 400:
            cv2.imwrite(
                os.path.join(save_dir, _fname),
                cv2.resize(image, (400, 400)),
            )
        else:
            cv2.imwrite(os.path.join(save_dir, _fname), image)

        cv2.imshow("image", image)
        k = cv2.waitKey(0)
        if k == ord("q"):
            break


def add_info_canvas(canvas, obj_name, label):
    _canvas = np.zeros_like(canvas)
    h, w, _ = _canvas.shape
    fontscale = np.ceil(h / 180) / 2
    thickness = max(round(fontscale), 1)
    (txt_w, txt_h), baseline = cv2.getTextSize(
        "Set End Point segmentation region", 1, fontscale, thickness
    )
    org_x = txt_h
    dy = txt_h + baseline + 5
    label_txt = ["negative", "positive"]

    cv2.putText(
        _canvas,
        f"set {obj_name} segmentation region",
        (org_x, txt_h + baseline),
        1,
        fontscale,
        (255, 0, 0),
        thickness + 1,
    )
    cv2.putText(
        _canvas,
        f"'0' : Label the point as negative",
        (org_x, dy * 3),
        1,
        fontscale,
        (0, 0, 255),
        thickness,
    )
    cv2.putText(
        _canvas,
        f"'1' : Label the point as positive",
        (org_x, dy * 2),
        1,
        fontscale,
        (0, 0, 255),
        thickness,
    )
    cv2.putText(
        _canvas,
        f"'p' : cancel(pop) last point",
        (org_x, dy * 4),
        1,
        fontscale,
        (0, 0, 255),
        thickness,
    )
    cv2.putText(
        _canvas,
        f"point label: {label_txt[label]}",
        (org_x, dy * 5),
        1,
        fontscale,
        (0, 255, 0),
        thickness,
    )

    canvas = cv2.hconcat([canvas, _canvas])

    return canvas, fontscale, thickness


def appoint(predictor: SAM2Stream, image, ann_frame_idx, obj_id, obj_name):
    cvi = cvInterface()
    win_name = "Generate Template"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, cvi.mouse_callback)

    label = 1
    labels = []
    points = []
    canvas = image.copy()
    h, w, _ = image.shape
    while True:
        canvas, fontscale, thickness = add_info_canvas(canvas, obj_name, label)
        cv2.imshow(win_name, canvas)
        k = cv2.waitKey(1)
        if k == ord("q"):  # quit
            break
        elif k == ord("0"):  # negative point
            label = 0
            print("label: ", label)
        elif k == ord("1"):  # positive point
            label = 1
            print("label: ", label)
        elif k == ord("p"):  # pop last point
            print("pop last point")
            labels.pop()
            points.pop()
            print(labels)
            print(points)

        if cvi.CLICKED:
            point = cvi.get_px()
            labels.append(label)
            points.append(point)
            print(labels)
            print(points)

        canvas = image.copy()
        if len(labels) > 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32),
            )

            # show the results on the current (interacted) frame
            mask = (
                (out_mask_logits[obj_id - 1] > 0.0).cpu().numpy().reshape(h, w)
            )  # boolean mask
            draw_mask(canvas, mask, color=(0, 0, 255))
            draw_points(canvas, np.array(points), np.array(labels), round(fontscale))
            draw_center_cross(
                canvas,
                mask.astype(np.uint8) * 255,
                round(fontscale * 1.5),
                label=obj_name,
            )

    return points, labels, mask


def set_roi(image):
    roi = None
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyAllWindows()
    
    print("roi", roi, "->")
    x, y, w, h = np.round(roi, decimals=-1).astype(np.int32)
    cx, cy = x + w // 2, y + h // 2
    l = max(w, h)
    x, y = cx - l // 2, cy - l // 2
    return x, y, l, l


def gen_template(
    predictor: SAM2Stream, camera: Union[ImageLoader, Basler], out_dir, roi=None
):
    """
    template_dir/
        L *.jpg : color image
        L 001/ : object_id -> object center
            L *.png : mask image
        L 002/ : object_id -> end point area
            L *.png : mask image
    """

    ### roi가 None 인 경우 설정 (crop 할 영역)
    if roi is None:
        _, image = camera.get()
        roi = set_roi(image.copy())
        print("roi", roi)
        np.savetxt(os.path.join(out_dir, "roi.txt"), roi)

    predictor.init()

    frame_segments = {}
    obj_ids = [1, 2]
    obj_class = ["Center", "End point"]
    h, w = roi[3], roi[2]
    fontscale = np.ceil(h / 180) / 2
    thickness = max(round(fontscale), 1)
    (txt_w, txt_h), baseline = cv2.getTextSize("iljpq", 1, fontscale, thickness)
    dy = txt_h + baseline + 5

    DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"
    idx = 0
    cv2.namedWindow("Generate Template", cv2.WINDOW_NORMAL)
    while True:
        ret, image = camera.get()
        print(type(image))
        if not ret:
            break
        cropped = image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        canvas = cropped.copy()
        cv2.putText(
            canvas,
            "'s': Select current frame",
            (txt_h, dy),
            1,
            fontscale,
            (0, 0, 255),
            thickness,
        )
        cv2.putText(
            canvas,
            "'q': Quit",
            (txt_h, dy * 2),
            1,
            fontscale,
            (0, 0, 255),
            thickness,
        )
        cv2.putText(
            canvas,
            "else any key",
            (txt_h, dy * 3),
            1,
            fontscale,
            (0, 0, 255),
            thickness,
        )

        cv2.imshow("Generate Template", canvas)
        k = cv2.waitKey(0)
        if k == ord("q"):
            break
        elif k == ord("s"):
            print("Generate Template")
            prompts = {}
            masks = {}
            frame_idx = predictor.add_conditioning_frame(cropped)
            print("--->", frame_idx)
            for obj_id in obj_ids:
                points, labels, out_boolean_mask = appoint(
                    predictor,
                    cropped,
                    frame_idx,
                    obj_id,
                    obj_class[obj_id - 1],
                )
                # out_mask = out_boolean_mask[obj_id - 1].astype(np.uint8) * 255
                out_mask = out_boolean_mask.astype(np.uint8) * 255
                draw_center_cross(
                    canvas, out_mask, thickness, label=obj_class[obj_id - 1]
                )

                prompts[obj_id] = points, labels
                masks[obj_id] = out_boolean_mask
                print(f"Set {obj_class[obj_id-1]} (id: {obj_id})")

            frame_segments[frame_idx] = {"prompts": prompts}
            # frame_segments[frame_idx]["mask"] = {s
            frame_segments[frame_idx]["mask"] = masks
            save_path = os.path.join(out_dir, f"{idx:05d}.jpg")
            cv2.imwrite(save_path, cropped)
            print("Save template to ", save_path)

            h, w, _ = cropped.shape
            for frame_idx, frame_info in frame_segments.items():
                per_obj_output_mask = frame_info["mask"]
                save_masks_to_dir(
                    output_mask_dir=out_dir,
                    frame_name=f"{idx:05d}.png",
                    per_obj_output_mask=per_obj_output_mask,
                    height=h,
                    width=w,
                    per_obj_png_file=True,
                    output_palette=DAVIS_PALETTE,
                )

            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Register model template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./sam/checkpoints/sam2.1_hiera_large.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="basler",
        help="inference input directory or camera",
    )
    parser.add_argument(
        "--template_dir",
        "-t",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )

    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="generate template mode. Save template masks to template_dir.",
    )

    parser.add_argument(
        "--roi",
        type=str,
        help="path to roi.txt",
    )

    parser.add_argument(
        "--test",
        help="test directory. save result images to <test>_results",
    )
    args = parser.parse_args()

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    #### Loading the SAM2 predictor
    # predictor = build_sam2_video_predictor(
    #     args.sam2_cfg, args.sam2_checkpoint, device=device
    # )
    predictor: SAM2Stream = build_sam2_stream_predictor(
        args.sam2_cfg, args.sam2_checkpoint, device=device
    )

    if args.source == "basler":
        # camera = Basler(device_class="BaslerGigE",ip_address="192.168.100.203")
        # camera.setConfig(exposureTime=80000.0)
        camera = Basler(device_class="BaslerGigE",ip_address="192.168.100.218")
        camera.setConfig(exposureTime=80000.0)
        # camera = Basler(model="acA5472-5gc")
        #camera.setConfig(exposureTime=80000)
    elif os.path.isdir(args.source):
        camera = ImageLoader(args.source)

    roi = None
    if args.roi:
        # roi: numpy array. [x, y, w, h]
        roi = np.loadtxt(args.roi).astype(np.int32)
        print(roi)

    if not os.path.exists(args.template_dir):
        os.makedirs(args.template_dir)
        print("Create directory: ", args.template_dir)
        if roi is not None:
            np.savetxt(os.path.join(args.tempalte_dir, "roi.txt"), roi)

    if args.generate:
        # generate template masks (cropped color image, mask image)
        gen_template(predictor, camera, args.template_dir, roi)

    if args.test:
        # calculate_angle(predictor, args.source, args.template_dir, save_dir)
        print("Loading templates ...")
        predictor.load_template_frames(args.template_dir)
        frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(args.template_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]

        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        template_masks = load_masks_from_source(
            args.template_dir, per_obj_png_file=True
        )
        for fname, template in template_masks.items():
            print(frame_names)
            if fname in frame_names:
                input_frame_idx = int(fname)
                print("input_frame_idx", input_frame_idx)
                per_obj_input_mask = template["per_obj_input_mask"]
                # input_palette = template["input_palette"]
                for object_id, object_mask in per_obj_input_mask.items():
                    predictor.add_new_mask(
                        frame_idx=input_frame_idx,
                        obj_id=object_id,
                        mask=object_mask,
                    )

        if args.test[-1] == "/" or args.test[-1] == "\\":
            args.test = args.test[:-1]
            
        save_dir = f"{args.test}_results"
        print("Save to ", save_dir)
        test(predictor, args.test, save_dir, roi)
