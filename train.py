import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np

import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tkinter import messagebox





window = tk.Tk()
def exitapp():
    msgbox=tk.messagebox.askquestion('QUIT','Are you sure you want to Quit',icon='warning')
    if msgbox=='yes':
        window.destroy()
window.title("Face Recognition Attendance System")
#bg=ImageTk.PhotoImage(file="me.jpg",master=window)
window.configure(background='#99bede')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
#Label1=Label(window,image=bg)
#Label1.place(x=0,y=0)
message = tk.Label(window, text="Face Recognition Attendance System" ,bg="#234c6f"  ,fg="white"  ,width=40  ,height=2,font=('times', 40, 'bold')) 
message.place(x=140, y=12)
lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="#234c6f"  ,bg="white" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=170)
txt = tk.Entry(window,width=20  ,bg="white" ,fg="#234c6f",font=('times', 15, ' bold '))
txt.place(x=700, y=195)
lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="#234c6f"  ,bg="white"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=250)
txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="#234c6f",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=270)
lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="#234c6f"  ,bg="white"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl3.place(x=400, y=330)
message = tk.Label(window, text="" ,bg="white"  ,fg="#234c6f"  ,width=35  ,height=2, activebackground = "silver" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=330)
lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="#234c6f"  ,bg="white"  ,height=2 ,font=('times', 15, ' bold')) 
lbl3.place(x=400, y=540)
message2 = tk.Label(window, text="" ,fg="#234c6f"   ,bg="white",activeforeground = "green",width=33  ,height=15  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=540)
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)
def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
def TakeImages():
    f=0       
    Id=(txt.get())
    name=(txt2.get())
    rows=[]
    with open("C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/StudentDetails/StudentDetails.csv") as csvf:
        re=csv.DictReader(csvf)
        for r in re:
            if Id==r['Id']:
                f=1
                break
            else:
                f=0
            
            
        
        
        
    csvf.close()
  
    

    if(is_number(Id) and name.isalpha() and f==0):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>29:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/StudentDetails/StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(f==1):
            res = "ID Already Exist"
            message.configure(text= res)
        elif(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        elif(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
        else:
            jjj=0
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/TrainingImageLabel/Trainner.yml")
    harcascadePath = "C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret,im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="C:/Users/T M RENUSHREE/Desktop/Face-Recognition-Based-Attendance-System-master/Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    
    fnn="Attendance_"+date+"_"+Hour+":"+Minute+":"+Second
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    print(attendance)
    res=attendance
    message2.configure(text= res)
    
    import smtplib
    from email.message import EmailMessage
    msg=EmailMessage()
    msg['Subject']=fnn
    msg['From']='Face Recognition Attendance System'
    msg['To']='facultycse1@gmail.com'
    with open (fileName,"rb")as f:
        file_data=f.read()
        file_name=f.name
        msg.add_attachment(file_data,maintype="application",subtype="csv",filename=fnn)  
    server=smtplib.SMTP_SSL('smtp.gmail.com',465)
    server.login("sattendance5@gmail.com","Asdf@1234")
    server.send_message(msg)
    server.quit()
    print("Mail sent!!")
    res = "Mail sent!!"
    message.configure(text= res)
  
    
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="#234c6f"  ,bg="white"  ,width=15  ,height=2 ,activebackground = "silver" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=165)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="#234c6f"  ,bg="white"  ,width=15  ,height=2, activebackground = "silver" ,font=('times', 15, ' bold '))
clearButton2.place(x=950, y=250)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="#234c6f"  ,bg="white"  ,width=20  ,height=3, activebackground = "silver" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=420)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="#234c6f"  ,bg="white" ,width=20  ,height=3, activebackground = "silver" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=420)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="#234c6f"  ,bg="white" ,width=20  ,height=3, activebackground ="silver",font=('times', 15, ' bold '))
trackImg.place(x=800, y=420)
quitWindow = tk.Button(window, text="Quit", command=exitapp  ,fg="#234c6f"  ,bg="white" ,width=20  ,height=3, activebackground ="silver" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=420)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
#mylist.pack(side=tk.LEFT)

window.mainloop()
