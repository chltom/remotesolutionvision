import cv2


class cvInterface:
    def __init__(self) -> None:
        self.x, self.y = 0, 0
        self.clicked_x, self.clicked_y = 0, 0
        self.CLICKED = False

    def mouse_callback(self, event, x, y, flags, param):
        self.x, self.y = x, y
        if event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            self.y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.CLICKED = True
            self.clicked_x = x
            self.clicked_y = y
            print(f"clicked ({x}, {y})")

    def get_px(self):
        _p = [self.clicked_x, self.clicked_y]
        self.CLICKED = False
        return _p

    def put_texts(
        self,
        img,
        txt_list,
        org,
        font,
        scale,
        thick,
        color=(0, 0, 255),
        border=False,
        upward=False,
    ):
        h, w = img.shape[:2]
        text_x = org[0]
        text_y = [0]
        text_w, text_h = 0, 0
        for txt in txt_list:
            (tw, th), baseline = cv2.getTextSize(txt, font, scale, thick)
            text_w = max(tw, text_w)
            text_h += th + baseline
            text_y.append(text_h)

        text_h += baseline
        if upward:
            text_y = np.array(text_y) + baseline
            text_y = org[1] - np.array(text_y[::-1])
        else:
            text_y = org[1] + np.array(text_y)

        margin = 5
        if text_x + text_w >= w:
            dx = w - (text_x + text_w)
            text_x += dx - margin

        if text_y[0] < margin:
            dy = margin + text_y[0]
            text_y += dy

        if text_y[0] + text_h >= h:
            dy = h - (text_y[0] + text_h)
            text_y += dy

        for txt, ty in zip(txt_list, text_y[1:]):
            if border:
                cv2.putText(
                    img, txt, (text_x, ty), font, scale, (255, 255, 255), thick + 4
                )
            cv2.putText(img, txt, (text_x, ty), font, scale, color, thick)
        return img
