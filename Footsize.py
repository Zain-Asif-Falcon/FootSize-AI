import torch
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import time
import cv2
from PIL import Image, ImageTk

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"PATH", force_reload=True)
model1 = tf.keras.models.load_model(r"C:\Users\Falcon\Downloads\size.h5")


from tkinter import *
root=Tk()
import PIL
# import cv2
root.title('Falcon IT prototype')
root.state("zoomed")
root.config(bg="#00406c")
# img = Image.open("path/to/image")
pic=PhotoImage(file=r"PATH")

frm=Label(root,image=pic,bg="#00406c")
frm.place(x=740,y=190)


l1=Label(frm,text='Welcome to FootXacles',font= ('Helvetica 15 bold'),bg="#00406c",fg="white")
l1.place(x=80,y=140)
l1=Label(frm,text='by Falcon IT',bg="#00406c",fg="white")
l1.place(x=150,y=170)

# ""


def camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Make detections 
        results = model(frame)
        
        cv2.imshow('YOLO', np.squeeze(results.render()))
        
        key = cv2.waitKey(10) & 0xFF

        # Press 'q' to quit
        if key == ord('q'):
            cv2.imwrite(r"C:\Users\Falcon\image.jpg", frame)
            print('Picture taken!')
            break

        
    cap.release()
    cv2.destroyAllWindows()

    size(r"C:\Users\Falcon\image.jpg")

bk=PhotoImage(file=r"C:\Users\Falcon\Downloads\icons8-back-arrow-48.png")


from tkinter import filedialog
def fun_1():
        
        file_path = filedialog.askopenfilename()
        cap = cv2.VideoCapture(file_path)
        
        while cap.isOpened():
            size(file_path)
           
            ret, frame = cap.read()
            
            # Make detections 
            results = model(frame)
            
            cv2.imshow('YOLO', np.squeeze(results.render()))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        


def fun():
    def back():
     frm11.destroy()
     frm1.destroy()
     frm.destroy()
    def fun_2():
        frm11.destroy()
        frm21=Label(root,image=pic,bg="#00406c")   
        frm21.place(x=740,y=190)
        ltm=Label(frm21 ,text="Please Choose the stage of the foot \n Camera Mode",font= ('Helvetica 11 bold'),bg="#00406c",fg="white")
        ltm.place(x=60,y=140)

        bb11=Button(frm21,image=b11,borderwidth=0,command=camera,bg="#00406c")
        bb11.place(x=100,y=240)
        bb21=Button(frm21,image=b22,borderwidth=0, command=camera,bg="#00406c"
                )
        bb21.place(x=230,y=240)
    def fun_3():
        frm11.destroy()
        
        frm21=Label(root,image=pic,bg="#00406c")  
        back1=Button(frm21,image=bk,borderwidth=0,command=back,bg="#00406c")
        back1.place(x=80,y=80) 
        frm21.place(x=740,y=190)
        ltm=Label(frm21 ,text="Please Choose the stage of the foot \n Explore Mode",font= ('Helvetica 11 bold'),bg="#00406c",fg="white")
        ltm.place(x=60,y=140)

        bb11=Button(frm21,image=b11,borderwidth=0,bg="#00406c", command= fun_1)
        bb11.place(x=100,y=240)
        bb21=Button(frm21,image=b22,borderwidth=0, command= fun_1,bg="#00406c"
                )
        bb21.place(x=230,y=240)
    frm.destroy()
    
   


frm1=Button(frm,image=pic1,borderwidth=0,command=fun,bg="#00406c")
frm1.place(x=160,y=400)

root.mainloop()
