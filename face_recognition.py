import cv2
import csv
import tkinter as tk
from PIL import Image
import os
import numpy as np

import datetime
import time

class tkinterApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self, bg='black')
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (HomePage, LoginPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            self.show_frame(HomePage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


    # Function for taking images of a new user
    # and saving them in TrainingImages folder
    def TakeImages(a, txt1, txt2):

        cam = cv2.VideoCapture(0)

        id = int(txt1.get())
        name = (txt2.get())

        # face_detector is a classifier that detects a face since it
        # has haarcascade features loaded onto it
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        print('Camera Opening. Focus On Camera')

        count = 0

        while True:

            # reading camera img and converting it to gray img
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # detectMultiScale is a pre-defined fun that helps
            # us to find the features/loc of the grey img
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # putting the face imgs in TrainingImages folder
                # cv2.imwrite(filename, image)-->syntax of imwrite
                cv2.imwrite('TrainingImages/Users.' + str(id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])

                count = count + 1
                cv2.imshow('Video Capture', img)

            if count >= 50:
                break
            k=cv2.waitKey(100) & 0xff
            if k==27:
                break
        cam.release()
        cv2.destroyAllWindows()

        # Make entry of the user into UserDetails.csv
        row = [id, name]

        with open(r'UserDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)

            writer.writerow(row)
        csvFile.close()

    def TrainImages(a):

        path = 'TrainingImages'
        # LBPH is a face recognition algo that extracts
        # image info and performs the matching
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces = []
        ids = []

        # func for getting the labels/id corr to each image
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                # Images are converted into Numpy Array in height, width, channel format
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(img_numpy)
                ids.append(id)

            return faces, ids

        print("Training faces. Wait a few seconds ...")
        faces, ids = getImagesAndLabels(path)
        # Saving the trained faces and their respective ID's
        # in a model named as "trainer.yml".
        recognizer.train(faces, np.array(ids))
        recognizer.save('trainer/trainer.yml')
        print("Number of faces trained : ", format(len(np.unique(ids))))

    def MorningAttendance(a):
        i = 1

        # This fun returns the name of the user by matching it with the id
        def UserDetails(id):
            i = 0
            f = open('UserDetails.csv')
            csv_f = csv.reader(f)
            for row in csv_f:
                for col in row:
                    if(col == str(id)):
                        i = 1
                        continue
                    if(i == 1):
                        return col

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            for(x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if (confidence < 75):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

                    id1 = id
                    id = str(id) + ' ' + UserDetails(id)
                    confidence = "  {0}%".format(round(100 - confidence))
                    if (i == 1):
                        row = [id1, UserDetails(id1), date, timeStamp]
                        with open(r"Attendance\MorningAttendance.csv", 'a+') as csvFile:
                            writer = csv.writer(csvFile)
                            # Entry of the row in csv file
                            writer.writerow(row)
                        csvFile.close()
                        i = i + 1
                else:
                    id = "Unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('Camera', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    def EveningAttendance(a):
        i = 1

        def UserDetails(id):
            i = 0
            f = open('UserDetails.csv')
            csv_f = csv.reader(f)
            for row in csv_f:
                for col in row:
                    if (col == str(id)):
                        i = 1
                        continue
                    if (i == 1):
                        return col

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        while True:

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if (confidence < 75):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

                    id1 = id
                    id = str(id) + ' ' + UserDetails(id)
                    confidence = "  {0}%".format(round(100 - confidence))
                    if (i == 1):
                        row = [id1, UserDetails(id1), date, timeStamp]
                        with open(r"Attendance\EveningAttendance.csv", 'a+') as csvFile:
                            writer = csv.writer(csvFile)
                            # Entry of the row in csv file
                            writer.writerow(row)
                        csvFile.close()
                        i = i + 1
                else:
                    id = "Unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                    # attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('Camera', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
# 75cfb8
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='white')

        msg = tk.Label(
            self, text="Online Attendance System",
            bg="#00af91", fg="white", width=55,
            height=3, font=('times', 30, 'bold'))
        msg.place(x=120, y=20)

        # lbl = tk.Label(self, text="Enter Your Details ",
        #                width=20, height=2, fg="#00af91",
        #                bg="white", font=('times', 15, ' bold '))
        # lbl.place(x=400, y=162)

        lbl1 = tk.Label(self, text="Employee Id : ",
                        width=20, height=2, fg="#00af91",
                        bg="white", font=('times', 15, ' bold '))
        lbl1.place(x=400, y=210)

        txt1 = tk.Entry(self,
                        width=20, bg="white",
                        fg="black", font=('times', 15, ' bold '))
        txt1.place(x=600, y=225)

        lbl2 = tk.Label(self, text="Employee_Name : ",
                        width=20, fg="#00af91", bg="white",
                        height=2, font=('times', 15, ' bold '))
        lbl2.place(x=400, y=260)

        txt2 = tk.Entry(self, width=20,
                        bg="white", fg="black",
                        font=('times', 15, ' bold '))
        txt2.place(x=600, y=275)

        MorningAttendance = tk.Button(self, text="Morning Attendance ",
                                      command=lambda:controller.MorningAttendance(), fg="white", bg="#00af91",
                                      width=15, height=2, activebackground="Red",
                                      font=('times', 15, ' bold '))
        MorningAttendance.place(x=380, y=360)

        EveningAttendance = tk.Button(self, text="Evening Attendance ",
                                      command=lambda:controller.EveningAttendance(), fg="white", bg="#00af91",
                                      width=15, height=2, activebackground="Red",
                                      font=('times', 15, ' bold '))
        EveningAttendance.place(x=650, y=359)

        takeImg = tk.Button(self, text="Sign Up ",
                            command=lambda: controller.show_frame(LoginPage), fg="white", bg="#00af91",
                            width=15, height=2, activebackground="Red",
                            font=('times', 15, ' bold '))
        takeImg.place(x=500, y=460)

        lbl = tk.Label(self, text="Instructions: If you are a new employee click on Sign Up otherwise mark the corresponding attendance",
                        width=100, fg="#00af91", bg="white",
                        height=2, font=('times', 15, ' bold '))
        lbl.place(x=250,y=550)


class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='white')

        message = tk.Label(
            self, text="Sign Up",
            bg="#00af91", fg="white", width=55,
            height=3, font=('times', 30, 'bold'))
        message.place(x=200, y=20)

        lbll = tk.Label(self, text="Enter Your Details ",
                        width=20, height=2, fg="#00af91",
                        bg="white", font=('times', 15, ' bold '))
        lbll.place(x=470, y=220)

        lbl1 = tk.Label(self, text="Employee Id : ",
                        width=20, height=2, fg="#00af91",
                        bg="white", font=('times', 15, ' bold '))
        lbl1.place(x=400, y=270)

        txt1 = tk.Entry(self, width=20, bg="white",
                        fg="black", font=('times', 15, ' bold '))
        txt1.place(x=600, y=285)

        lbl2 = tk.Label(self, text="Employee_Name : ",
                        width=20, fg="#00af91", bg="white",
                        height=2, font=('times', 15, ' bold '))
        lbl2.place(x=400, y=320)

        txt2 = tk.Entry(self, width=20,
                        bg="white", fg="black",
                        font=('times', 15, ' bold '))
        txt2.place(x=600, y=335)

        TakeImage = tk.Button(self, text="Take Images",
                              command=lambda: controller.TakeImages(txt1, txt2), fg="white", bg="#00af91",
                              width=15, height=2, activebackground="Red",
                              font=('times', 15, ' bold '))
        TakeImage.place(x=380, y=420)

        TrainImage = tk.Button(self, text="Train Images ",
                               command=lambda: controller.TrainImages(), fg="white", bg="#00af91",
                               width=15, height=2, activebackground="Red",
                               font=('times', 15, ' bold '))
        TrainImage.place(x=650, y=419)

        MainMenu = tk.Button(self, text="Return To Main Menu",
                             command=lambda: controller.show_frame(HomePage), fg='white'
                             , bg='#00af91', width=15, height=2, activebackground='red',
                             font=('times', 15, ' bold '))
        MainMenu.place(x=520, y=500)

        message1 = tk.Label(
            self, text="Instructions: Click on Take Images,your images will be clicked. After that click on Train Images.",
            bg='white', fg="#00af91", width=100,
            height=1, font=('times', 18, 'bold'))
        message1.place(x=0, y=580)

        message2 = tk.Label(
            self, text="Click on Return to Main Menu and mark your attendance.",
            bg="white", fg="#00af91", width=100,
            height=1, font=('times', 18, 'bold'))
        message2.place(x=0, y=610)


app = tkinterApp()
app.mainloop()
