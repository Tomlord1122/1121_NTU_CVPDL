from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from func import *
import os
import subprocess
import time
from tkinter import messagebox

win = Tk()
win.title("CVPDL Final")
win.geometry("1280x720")
win.configure(bg="#323232")
tag = 0
# 函数来读取图片

output_path = "../edge-connect/checkpoints/results/image1.png"
output_path2 = "../palette/palette_outputs/results/test/0/Out_image1.png"


def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        annotated_file_path = func(file_path)
        img = Image.open(annotated_file_path)
        # Resize the image to fill the screen
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.image = imgtk
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
def update_image():
    global tag
    if tag == 1 and os.path.isfile(output_path):
        img = Image.open(output_path)
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.image = imgtk
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
    elif tag == 2 and os.path.isfile(output_path2):
        img = Image.open(output_path2)
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.image = imgtk
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
def submit_func(): # edge-connect  tag = 1 
    # acquire the input text
    text = en.get()
    global tag
    tag = 1
    #step1: mask_path
    maskImg = Image.open(f"../edge-connect/examples/tom/maskCol/image1_{int(text) - 1}.png")
    maskImg_np = np.array(maskImg)
    cv2.imwrite(f"../edge-connect/examples/tom/masks/image1.png", maskImg_np)
    
    command = ["docker", "start", "edge-connect"]
    # 執行 Docker 命令
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e.stderr}")
    
    output_path = "../edge-connect/checkpoints/results/image1.png"
    while not os.path.exists(output_path):
        print("Waiting for image processing to complete...")
        time.sleep(5)  # Wait for 1 second before checking again

    if os.path.isfile(output_path):
        # Load and display the processed image
        imgtk = Image.open(output_path)
        canvas.image = ImageTk.PhotoImage(imgtk)
        canvas.create_image(0, 0, anchor=NW, image=canvas.image)
    else:
        messagebox.showerror("Error", "Processed image not found.")
    # imgtk = Image.open(f"../edge-connect/checkpoints/results/image1.png")
    # canvas.image = imgtk
    # canvas.create_image(0, 0, anchor=NW, image=imgtk)
    
    update_image()
    
def submit_func2(): # palette
    # acquire the input text
    text = en.get()
    global tag
    tag = 2
    global output_path2
    maskImg = Image.open(f"../palette/maskCol/image1_{int(text) - 1}.png")
    maskImg_np = np.array(maskImg)
    cv2.imwrite(f"../palette/masks/image1.png", maskImg_np)
    
    command = ["docker", "start", "palette"]
    # 執行 Docker 命令
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e.stderr}")
        
    output_path2 = f"../palette/palette_outputs/results/test/0/Out_image{text}.png"
    while not os.path.exists(output_path):
        print("Waiting for image processing to complete...")
        time.sleep(10)  # Wait for 1 second before checking again

    if os.path.isfile(output_path):
        # Load and display the processed image        
        img = Image.open(output_path)
        # Resize the image to fill the screen
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.image = imgtk
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
    else:
        messagebox.showerror("Error", "Processed image not found.")
        
    # imgtk = Image.open(f"../palette/palette_outputs/results/test/0/Out_image1.png")
    # canvas.image = imgtk
    # canvas.create_image(0, 0, anchor=NW, image=imgtk)  
    update_image()
    




    
                   
# 創建畫布和控制元素的框架
canvas = Canvas(win, bg="#323232")
control_frame = Frame(win, bg="#323232")

# 將畫布和框架放置到窗口
canvas.pack(side=LEFT, fill=BOTH, expand=True)
control_frame.pack(side=RIGHT, fill=Y)

# 在 control_frame 中創建和放置控制元素
promt = Label(control_frame, text="Please input what you want to delete", bg="#323232", fg="#ffffff")
promt.pack(pady=10)
en = Entry(control_frame, bg="#ffffff", fg="#000000")
en.pack(pady=10)
btn_open = Button(control_frame, text="Open Image", bg="#323232", fg="#ffffff", command=open_file)
btn_open.pack(pady=10)
btn_submit = Button(control_frame, text="model-edge", bg="#323232", fg="#ffffff", command=submit_func)
btn_submit.pack(pady=10)
btn_submit = Button(control_frame, text="model-palette", bg="#323232", fg="#ffffff", command=submit_func2)
btn_submit.pack(pady=10)

lbl = Label(control_frame, text="", bg="#323232", fg="#ffffff")
lbl.pack(pady=10)


# annotated_file_path = func(file_path)
# img = Image.open(annotated_file_path)
# # Resize the image to fill the screen
# img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
# imgtk = ImageTk.PhotoImage(image=img)
# canvas.image = imgtk
# canvas.create_image(0, 0, anchor=NW, image=imgtk)



if tag == 1 and os.path.isfile(output_path):
    img = Image.open(output_path)
    # Resize the image to fill the screen
    img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.image = imgtk
    canvas.create_image(0, 0, anchor=NW, image=imgtk)



if tag == 2 and os.path.isfile(output_path2): 
    img = Image.open(output_path2)
    # Resize the image to fill the screen
    img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.image = imgtk
    canvas.create_image(0, 0, anchor=NW, image=imgtk)


win.mainloop()