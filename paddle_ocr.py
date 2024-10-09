import torch
from paddleocr import PaddleOCR


class OcrBoxMask:
    def __init__(self):
        print("OcrFunction init")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
            {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'orc_box_mask'

    def orc_box_mask(self, images, text):
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
                            if text == "" or line[1][0].find(word) >= 0:
                                points = line[0]
                                # position to mask
                                mask[int(points[0][1]):int(points[2][1]), int(points[0][0]):int(points[1][0])] = 1
                                break
            masks.append(mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)


class OcrImageText:
    def __init__(self):
        print("OcrImageText init")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'orc_image_text'

    def orc_image_text(self, images):
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
