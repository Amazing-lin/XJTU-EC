import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
from torchvision import transforms as transforms
import os

# 设置图片保存路径
outfile = './out_pic'

# 创建一个界面窗口
win = tkinter.Tk()
win.title("picture process")
win.geometry("1280x1080")

# 设置全局变量
original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count = 0
img2 = tkinter.Label(win)


# 实现在本地电脑选择图片
def choose_file():
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    e.set(select_file)
    load = Image.open(select_file)
    load = transforms.Resize((300, 400))(load)
    # 声明全局变量
    global original
    original = load
    render = ImageTk.PhotoImage(load)

    img = tkinter.Label(win, image=render)
    img.image = render
    img.place(x=100, y=100)


# 随机比例缩放
def lessen():
    temp = original
    new_im = transforms.Resize((100, 200))(temp)
    print(f'{temp.size}---->{new_im.size}')
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im


# 随机位置裁剪
def random_cut():
    temp = original
    new_im = transforms.RandomCrop(100)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im


# 中心剪裁
def center_cut():
    temp = original
    new_im = transforms.CenterCrop(100)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im


# 随机水平翻转
def Horizon():
    temp = original
    new_im = transforms.RandomHorizontalFlip(p=1)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im


# 随机垂直翻转
def Vertical():
    temp = original
    new_im = transforms.RandomVerticalFlip(p=1)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im


# 随机角度旋转
def Rotation():
    temp = original
    new_im = transforms.RandomRotation(45)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im


# padding为正方形
def padding():
    temp = original
    new_im = transforms.Pad((0, (temp.size[0] - temp.size[1]) // 2))(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im


# 随机灰度化
def random_gray():
    temp = original
    new_im = transforms.RandomGrayscale(p=0.5)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im


# 设置亮度
def set_bright():
    def show_bright(ev=None):
        temp = original
        new_im = transforms.ColorJitter(brightness=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('亮度设置')
    scale = tkinter.Scale(top, from_=0, to=100, orient=tkinter.HORIZONTAL, command=show_bright)
    scale.set(1)
    scale.pack()


# 设置对比度
def set_contrast():
    def show_contrast(ev=None):
        temp = original
        new_im = transforms.ColorJitter(contrast=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('对比度设置')
    scale = tkinter.Scale(top, from_=0, to=100, orient=tkinter.HORIZONTAL, command=show_contrast)
    scale.set(1)
    scale.pack()


# 设置色度
def set_hue():
    def show_hue(ev=None):
        temp = original
        new_im = transforms.ColorJitter(hue=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('色度设置')
    scale = tkinter.Scale(top, from_=-0.5, to=0.5, resolution=0.1, orient=tkinter.HORIZONTAL, command=show_hue)
    scale.set(1)
    scale.pack()


# 设置饱和度
def set_saturation():
    def show_saturation(ev=None):
        temp = original
        new_im = transforms.ColorJitter(saturation=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('饱和度设置')
    scale = tkinter.Scale(top, from_=0, to=100, resolution=1, orient=tkinter.HORIZONTAL, command=show_saturation)
    scale.set(1)
    scale.pack()


# 保存函数
def save():
    global count
    count += 1
    save_img.save(os.path.join(outfile, 'test' + str(count) + '.jpg'))


# 显示路径
e = tkinter.StringVar()
e_entry = tkinter.Entry(win, width=68, textvariable=e)
e_entry.pack()

# 设置选择图片的按钮
button1 = tkinter.Button(win, text="Select", command=choose_file)
button1.pack()

# 设置标签分别为原图像和修改后的图像
label1 = tkinter.Label(win, text="Original Picture")
label1.place(x=200, y=50)

label2 = tkinter.Label(win, text="Modified Picture")
label2.place(x=900, y=50)

# 设置保存图片的按钮
button2 = tkinter.Button(win, text="save", command=save)
button2.place(x=600, y=100)

# 设置随机比例缩放的按钮
button3 = tkinter.Button(win, text="Random Lessen", command=lessen)
button3.place(x=600, y=150)

# 设置随机比例裁剪的按钮
button4 = tkinter.Button(win, text="Random Cut", command=random_cut)
button4.place(x=600, y=200)

# 设置center_cut按钮
button5 = tkinter.Button(win, text="Center Cut", command=center_cut)
button5.place(x=600, y=250)

# 设置随机水平翻转按钮
button6 = tkinter.Button(win, text="Random Horizontal Flip", command=Horizon)
button6.place(x=600, y=300)

# 设置随机垂直翻转按钮
button7 = tkinter.Button(win, text="Random Vertical Flip", command=Vertical)
button7.place(x=600, y=350)

# 设置随机角度旋转按钮
button8 = tkinter.Button(win, text="Random Rotation", command=Rotation)
button8.place(x=600, y=400)

# 设置padding按钮
button9 = tkinter.Button(win, text="Padding", command=padding)
button9.place(x=600, y=450)

# 设置随机灰度化按钮
button9 = tkinter.Button(win, text="Random Gray", command=random_gray)
button9.place(x=600, y=500)

# 设置亮度的按钮
button10 = tkinter.Button(win, text="Brightness", command=set_bright)
button10.place(x=600, y=550)

# 设置对比度的按钮
button11 = tkinter.Button(win, text="Contrast", command=set_contrast)
button11.place(x=600, y=600)

# 设置色度按钮
button12 = tkinter.Button(win, text="Hue", command=set_hue)
button12.place(x=600, y=650)

# 设置饱和度按钮
button13 = tkinter.Button(win, text="Saturation", command=set_saturation)
button13.place(x=600, y=700)

# 设置退出按钮
button0 = tkinter.Button(win, text="Exit", command=win.quit)
button0.place(x=600, y=750)
win.mainloop()
