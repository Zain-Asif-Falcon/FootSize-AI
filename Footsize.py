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


pic1=PhotoImage(file=r"E:\muneebfot\yolov5\icons8-next-page-64.png")
b1=PhotoImage(file=r"E:\muneebfot\yolov5\icons8-web-camera-48.png")
b2=PhotoImage(file=r"E:\muneebfot\yolov5\icons8-image-gallery-48.png")

b11=PhotoImage(file=r"C:\Users\Falcon\Downloads\icons8-foot-64.png")
b22=PhotoImage(file=r"C:\Users\Falcon\Downloads\icons8-trainers-64.png")

def size(file_path):
    path = file_path
    img = tf.keras.utils.load_img(
        path, target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model1.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['10', '11', '8', '9']


    print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score)))
    size11=Label(root, text="This image most likely belongs to {} \n with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score)),font= ('Helvetica 18 bold'),bg="#00406c",fg="white")
    size11.place(x=1160,y=340)





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
    
    frm11=Label(root,image=pic,bg="#00406c")   
    back1=Button(frm11,image=bk,borderwidth=0,command=back,bg="#00406c")
    back1.place(x=80,y=80)
    frm11.place(x=740,y=190)
    m =Label(frm11,text='MENU',font= ('Helvetica 25 bold'),bg="#00406c",fg='white')
    m.place(x=140,y=140)
    bb1=Button(frm11,image=b1,borderwidth=0,bg="#00406c",command=fun_2)
    bb1.place(x=100,y=240)
    bb2=Button(frm11,image=b2,borderwidth=0,bg="#00406c",command=fun_3
               )
    bb2.place(x=230,y=240)



   




frm1=Button(frm,image=pic1,borderwidth=0,command=fun,bg="#00406c")
frm1.place(x=160,y=400)







root.mainloop()
