import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import misc
import scipy
import skimage
from skimage import io,transform


def PIL_demo():
    # 1、读取图片
    img = Image.open('images/0(13)-0_SuperAIProExp_resize.bmp')

    # 2、显示图片/保存图片
    img.show()  # 展示图片
    img.save("images/PIL_save.jpeg")

    # 3、图片信息
    print(img.mode)  # 图像类型
    print(img.size)  # 图像的宽高

    # 4、图片操作

    # Image<->ndarray
    img_arr = np.array(img)  # 转为numpy形式，(H,W,C)
    new_img = Image.fromarray(img_arr)  # 再转换为Image形式

    # RGB->gray
    gray = Image.open('image.jpg').convert('L')  # 灰度图
    r, g, b = img.split()  # 通道的分离
    img = Image.merge('RGB', (r, g, b))  # 通道的合并

    img_copy = img.copy()  # 图像复制
    w, h = 64, 64
    img_resize = img.resize((w, h))  # resize
    img_resize.show()  # 展示图片

    # 剪切
    box = (200, 0, 500, 300)
    print(img)
    img2 = img.crop(box)
    plt.imshow(img2)
    plt.show()

    # 调整尺寸
    img2 = img.resize((400, 400))
    plt.imshow(img2)
    plt.show()
    # 左右对换。
    img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(img2)
    plt.show()
    # 上下对换。
    img2 = img.transpose(Image.FLIP_TOP_BOTTOM)
    plt.imshow(img2)
    plt.show()
    # 旋转 90 度角。注意只能旋转90度的整数倍
    img2 = img.transpose(Image.ROTATE_90)
    plt.imshow(img2)
    plt.show()

    # 颜色变换
    img2 = img.convert('1')  # 将图片转化为黑白
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('F')  # 将图片转化为32位浮点灰色图像
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('P')  # 将图片转化为 使用调色板映射到其他模式
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('RGB')  # 将图片转化为真彩
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('RGBA')  # 将图片转化为 真彩+透明
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('CMYK')  # 将图片转化为颜色隔离
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('YCbCr')  # 将图片转化为彩色视频格式
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('I')  # 将图片转化为32位整型灰色图像
    plt.imshow(img2)
    plt.show()
    img2 = img.convert('L')  # 将图片转化为黑白
    plt.imshow(img2)
    plt.show()


def Matplotlib_demo():
    # 1、读取图片
    img = plt.imread('images/2007_004476.png')  # 读取图片

    # 2、显示&保存
    plt.imshow(img)
    plt.imshow(img, cmap="gray")
    plt.show()
    # plt.savefig('images/Matplotlib_save.jpg')  # 保存图片
    #
    # # 3、图片信息
    # I = mpimg.imread('images/JPEG图像 2.jpeg')
    # print(I.shape)
    # plt.imshow(I)
    # plt.show()
    #
    # # 4、图片操作
    # img_r = img[:, :, 0]  # 灰度图
    # plt.imshow(img_r, cmap='Greys_r')  # 显示灰度图
    # plt.show()


def CV2_demo():
    # 1、读取图片
    img = cv2.imread('images/JPEG图像 2.jpeg')  # 读取图片

    # 2、显示&保存
    cv2.imshow('the window name', img)  # 显示图像
    cv2.waitKey()
    cv2.imwrite('images/cv_save.jpg', img)  # 保存图片

    # 3、图片信息
    print(type(img))  # 数据类型(numpy)
    print(img.dtype)  # 元素类型(uint8)
    print(img.shape)  # 通道格式(H,W,C)
    print(img.size)  # 像素点数

    # 4、图片操作
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR转灰度图
    gray = cv2.imread('images/cv_save.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图读取
    image = cv2.resize(img, (100, 200), interpolation=cv2.INTER_LINEAR)  # resize
    b, g, r = cv2.split(img)  # 通道分离
    merge_img = cv2.merge((b, g, r))  # 通道合并


def scipy_demo():
    # 1、读取图片
    I = scipy.misc.imread('images/JPEG图像 2.jpeg')

    # 2、显示&存储
    scipy.misc.imsave('images/scipy_save.jpg', I)   # 有错误
    plt.imshow(I)
    plt.show()


def skimage_demo():
    # 1、读取图片
    img = io.imread('images/JPEG图像 2.jpeg', as_gray=True)  # 读取图片 False原图，True灰度图

    # 2、显示图片
    # plt.imshow(img)
    # plt.show()
    # io.imshow(img)
    # io.show()
    io.imsave('images/skimage_save.jpg', img)

    # 3、图片信息
    print(type(img))  # 数据类型(numpy)
    print(img.dtype)  # 元素类型(uint8)
    print(img.shape)  # 通道格式(H,W,C)

    # 4、图片操作

    # 将图片的大小变为500x500
    img1 = transform.resize(img, (500, 500))

    # 缩小为原来图片大小的0.1
    img2 = transform.rescale(img, 0.1)

    # 缩小为原来图片行数一半，列数四分之一
    img3 = transform.rescale(img, [0.5, 0.25])

    # 放大为原来图片大小的2倍
    img4 = transform.rescale(img, 2)

    # 旋转60度，不改变大小
    img5 = transform.rotate(img, 60)

    # 旋转60度，同时改变大小
    img6 = transform.rotate(img, 60, resize=True)


if __name__ == "__main__":
    PIL_demo()
    # Matplotlib_demo()
    # CV2_demo()
    # scipy_demo()
    # skimage_demo()