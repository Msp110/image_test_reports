'''
    图像非线性变换之图像常规非线性变换
'''
#定义图像常规非线性变换函数
def Nonlinearity(img):#需要常规非线性变换的图像
    #获取图像属性包括高、宽、通道数
    h,w,c=img.shape
    #定义空白图像，用于存放图像常规非线性变换的结果，防止修改原图
    img1=np.zeros((h,w,c),dtype=img.dtype)
    #对原图进行遍历，通过图像常规非线性原理公式对图像进行常规非线性变换
    for i in range(h):
        for j in range(w):
            #获取原始图像个通道值并进行公式处理,然后进行防溢出处理
            b=min(255,int(int(img[i,j,0])*int(img[i,j,0])/255))
            g=min(255,int(int(img[i,j,1])*int(img[i,j,1])/255))
            r=min(255,int(int(img[i,j,2])*int(img[i,j,2])/255))
            #将常规非线性变换之后的值赋值给新图像
            img1[i,j]=[b,g,r]
    return img1
#导入函数库
import cv2
import numpy as np
#np.set_printoptions(threshold=np.inf)  #打印数组中的全部内容
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
#读取图像
img=cv2.imread("D://lena.jpg")
#调用图像非线性变换--常规非线性变换对图像进行处理
Nonlinearity=Nonlinearity(img)
#BGR转换为RGB显示格式，方便通过matplotlib进行图像显示
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
Nonlinearity=cv2.cvtColor(Nonlinearity,cv2.COLOR_BGR2RGB)
#图像与原图显示对比
titles = [ '原图像', '图像非线性变换之常规非线性变换']
images = [img, Nonlinearity]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')#关闭坐标轴  设置为on则表示开启坐标轴
plt.show()
