from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import PyPDF2
import subprocess
import webbrowser 
import mediapipe as mp
from tensorflow.keras.models import load_model
mp_holistic = mp.solutions.holistic #(holistic model) 
mp_drawing = mp.solutions.drawing_utils
model = load_model('action.h5')

 
# sets the window's size and colour 
root = Tk() 
root.title('ASL')
root.geometry('700x700')
root.configure(bg='midnight blue')

mylabel3 = Label(root, text = '                                ', bg='midnight blue').grid(row=0,column=150)
mylabel1 = Label(root, text='Real time sign langauge detection.', bg='midnight blue', fg='white', font=('Eras Bold ITC' ,24,'bold', 'italic')).grid(row=10, column = 10)
mylabel2= Label(root, text='Here you can detect and learn some ASL gestures!', bg='midnight blue', fg='white',font=('Eras Bold ITC',20,'bold')).grid(row = 20, column = 10)



def image1():  #opens a picture of the american sign language to practice and refer to 
   
      global img
      canvas = Canvas(root, width = 290, height = 250, bg = 'midnight blue', highlightbackground = 'midnight blue') 
      img = ImageTk.PhotoImage(Image.open("ASL1.png"))  
      canvas.create_image(20, 20, anchor=NW, image=img)
      canvas.grid(row = 500, column = 10)

      button1 = Button(root, text="Close", font=('Gadugi', 5, 'bold'), fg='white', bg='turquoise', command=canvas.destroy)
      button1.grid(row=700, column=10)
   


    

def detector (): #detects ASL gestures real time 
   
   def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False    
    results = model.process(image)   
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
   def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #  left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #  right hand connections
   def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(80,200,20), thickness=1, circle_radius=1),mp_drawing.DrawingSpec(color=(37,37,37),thickness=1,circle_radius =1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius =2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius =2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius =2))
   def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

   actions = np.array(['hello', 'thanks', 'sorry', 'yes', 'no', 'goodbye', 'please', ' ']) #all the symbols currently detected 



   sequence = []
   sentence = []
   predictions = []
   threshold = 0.8

   cap = cv2.VideoCapture(0)
   

   with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic: #sets parameters for detection and displaying the detected symbol
       while cap.isOpened():

           
        #reads the camera
           ret, frame = cap.read()

        # Make detections
           image, results = mediapipe_detection(frame, holistic)
         
        
           draw_styled_landmarks(image, results)
        
        
           keypoints = extract_keypoints(results)
           sequence.append(keypoints)
           sequence = sequence[-20:]

            
           
           if len(sequence) == 20:
               res = model.predict(np.expand_dims(sequence, axis=0))[0]
                
               predictions.append(np.argmax(res))

               #predictions to minimise false detection 
    
               if np.unique(predictions[-20:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

           if len(sentence) > 5: 
                sentence = sentence[-5:]
               
           cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
           cv2.putText(image, ' '.join(sentence), (3,30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
           
           
           cv2.imshow('OpenCV Feed', image)

           # Breaks feed
           if cv2.waitKey(10) & 0xFF == ord('q'):
               break
       cap.release()
       cv2.destroyAllWindows()
   
def user_manual ():
    
   def callback(event):
    webbrowser.open_new(event.widget.cget("text")) #displays a link to the user manual 

      
   lbl = Label(root, text=r"https://tinyurl.com/2a6hspjy", fg="white", cursor="hand2", bg='turquoise')
   lbl.grid(row = 310, column = 10)
   lbl.bind("<Button-1>", callback)
   root.mainloop()
   
def About_help (): #gives basic information about this application 
   text1 = Label(root, text='This is an ASL detector, and can be used to practice and train yourself, or to communicate with someone. \n It uses a camera and hand movement to make these predictions. Enjoy! \n\n If you have any queries or complaints, please contact me here: riddhichauhan2005@gmail.com \nI will try to get back to you as soon as possible.', bg='Navy', fg='white', font=('Century Gothic', 10))
   text1.grid(row=570, column = 10)
   button = Button(root, text="Close", font=('Gadugi', 5, 'bold'), fg='white', bg='turquoise', command=text1.destroy)
   button.grid(row=980, column=10)
   
mylabel4 = Label(root, text = '                                ', bg='midnight blue').grid(row=50,column=10)

About = Button(root, text=' About and contacts  ', fg=('firebrick1'), bg=('lemon chiffon'), command=About_help, font=('Gadugi', 14, 'bold')).grid(row =400, column =10)

mylabel4 = Label(root, text = '                                ', bg='midnight blue').grid(row=350,column=10)

user_manual1 = Button(root, text = '       User manual        ', fg=('firebrick1'), command=user_manual, bg=('lemon chiffon'), font=('Gadugi', 14, 'bold')).grid(row =300, column =10)

mylabel = Label(root, text = '                                ', bg='midnight blue').grid(row=250,column=10)
   
image1 = Button(root, text='          Symbols           ', command=image1, fg=('firebrick1'), bg=('lemon chiffon'), font=('Gadugi', 14, 'bold')).grid(row=200, column =10)

mylabel = Label(root, text = '                                ', bg='midnight blue').grid(row=150,column=10)

detector1 = Button(root, text='       Detect here        ', fg=('firebrick1'), bg=('lemon chiffon'), font=('Gadugi', 14, 'bold'), command=detector).grid(row =100, column =10)




root.mainloop() 
