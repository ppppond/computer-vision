from PIL import Image, ImageEnhance

def transform_image(img, rotate, flip_h, flip_v, bright, cont, col):
    if rotate != 0:
        img = img.rotate(rotate, expand=True)
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    img = ImageEnhance.Brightness(img).enhance(bright)
    img = ImageEnhance.Contrast(img).enhance(cont)
    img = ImageEnhance.Color(img).enhance(col)
    return img
