# Hybrid-RFID-Face-Attendance-System

copy in drive IOT & EM



This project is a **RFID + Face Recognition-based Automated Attendance System**.
It uses **OpenCV, SQLite, and Python** to detect faces, recognize students, and mark their attendance automatically.

---

## Tools & Requirements

* **VS Code** (Recommended IDE)
* **Python 3.9.6** (Recommended)
     * Works with Python 3.13 also (make sure pip packages install properly)
* **SQLite** (Database)
* **RFID Module with tag

---

## Installation & Setup

### Step 1: Download the Project

* Click **Code → Download ZIP** from GitHub.
* Extract the ZIP file into your Laptop/PC.

### Step 1.1 : RFID_Reader_Module_code download
   * Open Arduino IDE Open this file Code in sketch and install this library MFRC522 in your arduino IDE
   * upload this code in Arduino Board Scan a tag and Note or Copy a Tag ID  Like ( TAG ID : 4A00CB7D6894 )
   * Next Step close Arduino IDE open VSCode also follow below step but dont diconnect USB Arduino Cable in Arduino Board in Labtap 
   * After python code Run , Enroll time Enter a TAG ID which student data store datavase
   * Recognition TAG ID and FACE

### Step 2: Open in VS Code

* Open **VS Code** → Click **Open Folder** → Select the extracted project folder.

### Step 3: Install Required Packages

Run the following command in VS Code Terminal:
```bash
pip install flask
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install pandas
pip install pillow
pip install sqlite3-binary   # sometimes sqlite3 needs manual install
pip install pickle5          # only if pickle errors occur
```
or Simply install required package used this command
```bash
pip install -r requirements.txt
```


### Step 4: Run the Main Program

The main script is main.py. It accepts different commands for enrollment, training, recognition, and exporting attendance.

Usage:
python main.py [-h] (--enroll | --train | --recognize | --export)

 like terminal is :
 ```bash
   usage: main.py [-h] (--enroll | --train | --recognize | --export)
   main.py: error: one of the arguments --enroll --train --recognize --export is required
   C:\Users\hp\Downloads\face2>
```
 ---------------------------------------------
  for example:
  
  C:\Users\hp\Downloads\face2> python main.py --enroll
  Others commands follow below


## How to Use

### 1. Enroll Your Face in Database
```bash
python main.py --enroll
```

* Captures your face and saves images in the `dataset/` folder.
* Stores student details in the database (`attendance.db`).



### 2. Train Your Face Images
```bash
python main.py --train
```
* Trains the face recognition model.
* Saves trained data into `trainer.yml`.

### 3. Recognize Faces & Mark Attendance
```bash
python main.py --recognize
```

* Detects and recognizes faces through the webcam.
* Automatically marks attendance in `attendance.db`.
* If a student’s attendance is already marked, it will indicate so.

---

### 4. Export Attendance to CSV
```bash
python main.py --export
```

* Exports recent attendance records from the database into `attendance_log.csv`.



## Web Interface

This project also includes a **basic web page**.

Run:

python apps.py

* The `templates/` folder contains the HTML files:

  * **index.html** → Homepage
  * **enroll.html** → Student Enrollment form

---

## Database

* When running `main.py`, a database file `attendance.db` is automatically created.
* To view it in VS Code, install the **SQLite Viewer Extension**.

---

## Helpful Commands

File: `cmt_code_run.txt`

* This file contains frequently used commands for quick reference:

```bash
python main.py --enroll
python main.py --train
python main.py --recognize
python main.py --export
```

---

## 🔧 Notes

* Some extra files in the project are optional and not required for normal usage.
* Make sure your camera is working properly before running `--recognize`.

---


## 6th also other is used for obtional only not need 

__________________________________________________________
1. Face Detection

Algorithm Used: Haar Cascade Classifier

File:

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)


Purpose: Detects where the face is in the image.

Haar cascades use machine learning with features + AdaBoost, very fast for real-time webcam.

2. Face Recognition

****Algorithm Used: LBPH (Local Binary Patterns Histograms)*****

Code part:

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
label, confidence = recognizer.predict(face_resized)


Purpose: Identifies whose face it is.

How LBPH works:

Converts image to grayscale.
Divides image into small grids.
For each pixel, compares with neighbors → builds a binary pattern.
Builds histograms for all grids → combines them.
Compares histograms of new face with trained dataset.

Confidence value:
Lower = better match.
In your code:
CONFIDENCE_THRESHOLD = 60.0
Faces with confidence < 60 are accepted as “recognized”.


Summary:

## Face Detection → Haar Cascade Classifier

## Face Recognition → LBPH (Local Binary Patterns Histogram)
       
                
   



