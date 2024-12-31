# ComfyUI-PaddleOcr
Nodes related to PaddleOCR OCR
- Inspire by [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/) 


## OcrBoxMask
### return masks of detected text in images
This node returns masks of detected text in images using PaddleOCR.
### Inputs
- `images`: the input images to be processed.
-  text: the text to be detected in images.
### Outputs
- `masks`: the masks of detected text in the input images.

## OcrImageText
### return text in an image
This node returns text in an image using PaddleOCR.
### Inputs
- `image`: the input image to be processed.
### Outputs
- `text`: the text detected in the input image.
