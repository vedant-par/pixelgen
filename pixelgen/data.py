import os
from tqdm import tqdm
from torchvision import transforms as T
from PIL import ImageOps, Image

size = 40
pixel_list = []
x1 = size//4
y1 = x1*3

def get_data():
    for img in tqdm(os.listdir("data/toons")):
        img = Image.open(f"data/toons/{img}")
        transform = T.Resize((size, size))
        img = transform(img)
        img = ImageOps.grayscale(img)
        img = img.crop((x1, x1, y1, y1))
        pixel_list.append(list(img.getdata()))

    length, width = img.size
    return pixel_list, length

if __name__ == "__main__":
    get_data()
