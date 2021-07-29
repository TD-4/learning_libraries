import cv2
import os


def jpg2bmp(jpg_dir="", bmp_dir=""):
    jpg_images = os.listdir(jpg_dir)
    for jpg_p in jpg_images:
        img = cv2.imread(os.path.join(jpg_dir, jpg_p), -1)
        newName = jpg_p.replace('.jpg', '.bmp')
        cv2.imwrite(os.path.join(bmp_dir, newName), img)


if __name__ == "__main__":
    jpg2bmp(jpg_dir=r"E:\test", bmp_dir=r"E:\output")