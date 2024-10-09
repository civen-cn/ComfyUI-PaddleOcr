import kornia
import torch
from paddleocr import PaddleOCR
import comfy.model_management


class OcrBoxMask:
    def __init__(self):
        print("OcrFunction init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {"required":
            {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'orc_box_mask'

    def orc_box_mask(self, images, text, lang):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        masks = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            shape = i.shape
            mask = torch.zeros((shape[0], shape[1]), dtype=torch.uint8)
            words = text.split(";")
            result = self.ocr.ocr(i, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        # print(line[1][0])
                        for word in words:
                            if word == "":
                                continue
                            if text == "" or line[1][0].find(word) >= 0:
                                text_line = line[1][0]
                                points = line[0]
                                if points[0][1] > 1:
                                    points[0][1] -= 1
                                if points[2][1] < shape[0] - 1:
                                    points[2][1] += 1
                                total_length = len(text_line)
                                start = 0
                                while text_line.find(word, start) >= 0:
                                    start = text_line.find(word, start)
                                    end = start + len(word)
                                    x_min = points[0][0] + start * (points[1][0] - points[0][0]) / total_length
                                    x_max = points[0][0] + end * (points[1][0] - points[0][0]) / total_length
                                    if x_min > 1:
                                        x_min -= 1
                                    if x_max < shape[1] - 1:
                                        x_max += 1

                                    mask[int(points[0][1]):int(points[2][1]), int(x_min):int(x_max)] = 1
                                    start = end
            masks.append(mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)


class OcrImageText:
    def __init__(self):
        print("OcrImageText init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {
            "required": {
                "images": ("IMAGE",),
                "lang": (lang_list, {"default": "ch"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'orc_image_text'

    def orc_image_text(self, images, lang):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

        text = ""
        last_text = ""
        for image in images:
            i = 255. * image.cpu().numpy()
            now_text = ""
            orc_ret = self.ocr.ocr(i, cls=False)
            for idx in range(len(orc_ret)):
                res = orc_ret[idx]
                if res is not None:
                    for line in res:
                        if line[1][0] != "":
                            now_text += line[1][0] + "\n"
            if now_text != "" and now_text != last_text:
                text += now_text + "\n"
                last_text = now_text
        return (text,)


class OcrBlur:
    def __init__(self):
        print("OcrBlur init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {"required":
            {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
                "blur": ("INT", {"default": 255, "min": 3, "max": 8191, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'orc_blur'

    def orc_blur(self, images, text, lang, blur):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        new_images = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            shape = i.shape
            mask = torch.zeros((shape[0], shape[1]), dtype=torch.uint8)
            words = text.split(";")
            result = self.ocr.ocr(i, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        # print(line[1][0])
                        for word in words:
                            if word == "":
                                continue
                            if text == "" or line[1][0].find(word) >= 0:
                                text_line = line[1][0]
                                points = line[0]
                                if points[0][1] > 1:
                                    points[0][1] -= 1
                                if points[2][1] < shape[0] - 1:
                                    points[2][1] += 1
                                total_length = len(text_line)
                                start = 0
                                while text_line.find(word, start) >= 0:
                                    start = text_line.find(word, start)
                                    end = start + len(word)
                                    x_min = points[0][0] + start * (points[1][0] - points[0][0]) / total_length
                                    x_max = points[0][0] + end * (points[1][0] - points[0][0]) / total_length
                                    if x_min > 1:
                                        x_min -= 1
                                    if x_max < shape[1] - 1:
                                        x_max += 1

                                    mask[int(points[0][1]):int(points[2][1]), int(x_min):int(x_max)] = 1
                                    start = end

            # blur the image by mask
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.permute(0, 3, 1, 2)
            blurred = image.clone()
            alpha = mask_floor(mask_unsqueeze(mask))
            alpha = alpha.expand(-1, 3, -1, -1)
            blurred = gaussian_blur(blurred, blur, 0)
            blurred = image + (blurred - image) * alpha
            new_images.append(blurred.permute(0, 2, 3, 1))
        return (torch.cat(new_images, dim=0),)


def gaussian_blur(image, radius: int, sigma: float = 0):
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    image = image.to(comfy.model_management.get_torch_device())
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma)).cpu()


def mask_floor(mask, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


def mask_unsqueeze(mask):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask