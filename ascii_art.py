import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFont, ImageDraw


class Masker:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self.transform = torchvision.transforms.ToTensor()

    def extract_mask(self, img):
        self.model.eval()

        # use mask-rcnn to extract mask
        trans_img = self.transform(img)
        outputs = self.model([trans_img])

        # make binary mask
        mask: torch.Tensor = outputs[0]['masks'] > 0.5
        # 1 for keep and 0 for discard during multiplication
        mask_image = torch.where(mask.squeeze(1), 1.0, 0.0)
        # collapse all mask to a single 2-D mask tensor by summation
        mask_image = torch.sum(mask_image, dim=0)[None, :]
        mask_image = torch.where(mask_image >= 1.0, trans_img, 0.0)

        # return proper image value range and dimension
        result = (mask_image * 255).type(torch.uint8).permute(1, 2, 0).numpy()

        return result


class ASCII_Conversion:
    def __init__(self):
        self.density = """ .-=^+*#%$&@"""
        self.pixel_max = 255.0
        self.down_ratio = 0.1

    def pixel_to_ascii(self, v):
        # convert the range into a 0-1 range
        scaled = float(v) / self.pixel_max
        # convert the 0-1 range into a value in the right range.
        idx = int(scaled * (len(self.density) - 1))
        return self.density[idx]

    def image_to_text(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        down_shape = [int(s * self.down_ratio) for s in img.shape]
        img = cv2.resize(img, down_shape)
        m = np.array(img)
        text = ''
        for i in range(m.shape[0]):
            for j in range(m.shape[1] - 1, -1, -1):
                text += f' {self.pixel_to_ascii(m[i][j])} '
            text += '\n'
        return text

    def text_to_monospaced_image(self, text, shape):
        # make blank image
        gray_image = Image.fromarray(np.full(shape, 0, dtype=np.uint8))

        # get a drawing context
        draw = ImageDraw.Draw(gray_image)
        monospace = ImageFont.truetype(r"./fonts/andalemo.ttf", 12)
        draw.text(xy=(0, 0), text=text, fill=255, font=monospace)

        # convert back to image
        result_o = np.array(gray_image)
        return result_o

    def run(self, img):
        text = self.image_to_text(img)
        img = self.text_to_monospaced_image(text, (1000, 1000))
        return img


if __name__ == '__main__':
    # initialize the camera
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    # initialize main handler
    ascii_conversion = ASCII_Conversion()

    # initialize masker
    masker = Masker()

    while 1:
        _, image = cam.read()

        image = masker.extract_mask(image)
        image = ascii_conversion.run(image)

        cv2.imshow("Camera", image)

        # if keyboard interrupt occurs
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

