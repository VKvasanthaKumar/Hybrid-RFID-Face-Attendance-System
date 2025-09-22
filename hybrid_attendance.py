# hybrid_attendance.py
import os
import cv2
import numpy as np
import sqlite3
import argparse
import pickle
from datetime import datetime, date, timedelta
import pandas as pd
import threading
import queue
import time
import serial  # pip install pyserial

# ---------- CONFIG ----------
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
DB_FILE = "attendance.db"
ATTENDANCE_CSV = "attendance_log.csv"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_PER_PERSON = 30
IMG_WIDTH, IMG_HEIGHT = 200, 200
CONFIDENCE_THRESHOLD = 60.0

# RFID / Serial config
SERIAL_PORT = "COM27"        # change to your Arduino serial port (e.g., "/dev/ttyUSB0" on Linux)
SERIAL_BAUD = 115200
TAG_WAIT_SECONDS = 10        # how long to wait for face after tag scan
TAG_DEBOUNCE_SECONDS = 3    # ignore repeated tag reads within this many seconds
# ----------------------------

# Thread-safe queue for incoming tags
tag_queue = queue.Queue()
last_tag_times = {}  # tag -> last seen timestamp to debounce

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            reg_no TEXT UNIQUE,
            name TEXT,
            rfid_tag TEXT UNIQUE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            timestamp TEXT,
            method TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_student_db(name, reg_no, rfid_tag=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (reg_no, name, rfid_tag) VALUES (?, ?, ?)", (reg_no, name, rfid_tag))
        conn.commit()
        student_id = c.lastrowid
    except sqlite3.IntegrityError:
        # reg_no or tag exists -> get id
        if rfid_tag:
            c.execute("SELECT id FROM students WHERE rfid_tag = ?", (rfid_tag,))
            row = c.fetchone()
            if row:
                student_id = row[0]
            else:
                c.execute("SELECT id FROM students WHERE reg_no = ?", (reg_no,))
                row = c.fetchone()
                student_id = row[0]
        else:
            c.execute("SELECT id FROM students WHERE reg_no = ?", (reg_no,))
            row = c.fetchone()
            student_id = row[0]
    conn.close()
    return student_id

def get_students():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, reg_no, name, rfid_tag FROM students")
    rows = c.fetchall()
    conn.close()
    return rows

def get_student_by_tag(tag):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, reg_no, name FROM students WHERE rfid_tag = ?", (tag,))
    row = c.fetchone()
    conn.close()
    return row  # (id, reg_no, name) or None

def mark_attendance(student_id, name, method="face"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    # Check last attendance for the student today
    c.execute("SELECT timestamp FROM attendance WHERE student_id = ? ORDER BY timestamp DESC LIMIT 1", (student_id,))
    row = c.fetchone()
    if row:
        last_ts = datetime.fromisoformat(row[0])
        if last_ts.date() == date.today():
            conn.close()
            return False  # already marked today
    c.execute("INSERT INTO attendance (student_id, name, timestamp, method) VALUES (?, ?, ?, ?)",
              (student_id, name, now, method))
    conn.commit()
    conn.close()
    # append to CSV
    df_row = pd.DataFrame([{"student_id": student_id, "name": name, "timestamp": now, "method": method}])
    if not os.path.exists(ATTENDANCE_CSV):
        df_row.to_csv(ATTENDANCE_CSV, index=False)
    else:
        df_row.to_csv(ATTENDANCE_CSV, mode='a', header=False, index=False)
    return True

def enroll_student(name, reg_no, rfid_tag=None, serial_port=None):
    student_id = add_student_db(name, reg_no, rfid_tag)
    folder_name = f"{student_id}_{reg_no}_{name.replace(' ', '_')}"
    path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(path, exist_ok=True)

    # If no rfid_tag provided, you can scan it from serial interactively
    if rfid_tag is None and serial_port:
        print("Waiting for RFID tag on serial. Scan tag now...")
        # block until tag appears on serial
        with serial.Serial(serial_port, SERIAL_BAUD, timeout=10) as ser:
            start = time.time()
            found = False
            while time.time() - start < 15:  # 15s wait
                line = ser.readline().decode(errors='ignore').strip()
                if line.startswith("TAG:"):
                    rfid_tag = line.split("TAG:")[-1].strip()
                    print("Captured tag:", rfid_tag)
                    found = True
                    break
            if not found:
                print("No tag found on serial. Continue without tag.")
    # update student record if we got a tag
    if rfid_tag:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("UPDATE students SET rfid_tag = ? WHERE id = ?", (rfid_tag, student_id))
            conn.commit()
        except Exception as e:
            print("Failed to update tag:", e)
        conn.close()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print(f"[ENROLL] Capturing {IMG_PER_PERSON} face images for {name} (ID {student_id}). Press 'q' to quit early.")
    count = 0
    while count < IMG_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            img_path = os.path.join(path, f"{str(count).zfill(3)}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/{IMG_PER_PERSON}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            break
        cv2.imshow("Enroll - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images into {path}. Run `--train` to train the model.")

def train_model():
    print("[TRAIN] Scanning dataset and training LBPH recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    curr_label = 0

    for person_folder in os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else []:
        folder_path = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(folder_path): continue
        try:
            student_id_str, reg_no, name = person_folder.split("_", 2)
            student_id = int(student_id_str)
        except Exception:
            student_id = curr_label + 1
            name = person_folder
        label = student_id
        label_map[label] = f"{name} ({reg_no})" if '_' in person_folder else name
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            faces.append(img)
            labels.append(label)
        curr_label = max(curr_label, label)

    if not faces:
        print("No training data found. Enroll students first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.write(TRAINER_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)
    print(f"[TRAIN] Training complete. Saved {TRAINER_FILE} and {LABELS_FILE}.")

# Serial listening thread: reads lines like "TAG:44AF3C21" and puts into queue
def serial_thread_fn(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print("Serial open error:", e)
        return
    print("[SERIAL] Listening for tags on", port)
    while True:
        try:
            raw = ser.readline().decode(errors='ignore').strip()
            if not raw:
                continue
            if raw.startswith("TAG:"):
                tag = raw.split("TAG:")[-1].strip()
                now = time.time()
                # debounce by tag
                last = last_tag_times.get(tag)
                if last and (now - last) < TAG_DEBOUNCE_SECONDS:
                    continue
                last_tag_times[tag] = now
                print("[SERIAL] Tag detected:", tag)
                tag_queue.put((tag, now))
        except Exception as e:
            print("Serial read error:", e)
            break
    ser.close()
    return tag

def recognize_loop(serial_port=None):
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        print("No trained model found. Run `--train` first.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Start serial thread if port provided
    if serial_port:
        t = threading.Thread(target=serial_thread_fn, args=(serial_port, SERIAL_BAUD), daemon=True)
        t.start()

    expected_tag = None
    expected_student = None
    expected_until = None

    print("[RECOGNIZE] Press 'q' to quit.")
    while True:
        # check tag queue
        try:
            tag, tstamp = tag_queue.get(block=False)
            student = get_student_by_tag(tag)
            if student:
                student_id = student[0]
                student_name = student[2]
                expected_tag = tag
                expected_student = student_id
                expected_until = datetime.now() + timedelta(seconds=TAG_WAIT_SECONDS)
                print(f"[RFID] Expecting face for {student_name} (ID {student_id}) until {expected_until}")

            else:
                # Unknown tag -> log it as unknown (optional)
                print(f"[RFID] Unknown tag scanned: {tag}")
                expected_tag = tag
                expected_student = None
                expected_until = datetime.now() + timedelta(seconds=TAG_WAIT_SECONDS)
        except queue.Empty:
            pass

        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        face_displayed = False
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            label, confidence = recognizer.predict(face_resized)
            if confidence < CONFIDENCE_THRESHOLD:
                display = label_map.get(label, f"ID:{label}")
                try:
                    student_id = int(label)
                except Exception:
                    student_id = None

                # If a tag was scanned and we expect a student, check match
                if expected_student is not None and expected_until and datetime.now() <= expected_until:
                    if student_id == expected_student:
                        marked = mark_attendance(student_id, display, method="rfid+face")
                        msg = "Present (RFID+Face)" if marked else "Already Present"
                        color = (0,255,0)
                        # Clear expectation
                        expected_student = None
                        expected_tag = None
                        expected_until = None
                    else:
                        # Face does not match tag
                        marked = mark_attendance(expected_student, "MISMATCH", method="rfid_no_face")
                        msg = "Tag scanned but face mismatch -> Logged absent/mismatch"
                        color = (0,165,255)
                        expected_student = None
                        expected_tag = None
                        expected_until = None
                else:
                    # No recent tag expectation -> normal face-only marking
                    if student_id is not None:
                        marked = mark_attendance(student_id, display, method="face")
                        if marked:
                            msg = "Present (face-only)"
                            color = (0,255,0)
                        else:
                            msg = "Already Present"
                            color = (0,165,255)
                    else:
                        msg = "Recognized but no ID"
                        color = (0,255,255)

                text = f"{display} ({confidence:.1f}) - {msg}"
            else:
                text = f"Unknown ({confidence:.1f})"
                color = (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            face_displayed = True
            break

        # If a tag expectation expired and no matching face arrived -> mark absent/mismatch
        if expected_until and datetime.now() > expected_until:
            if expected_student:
                print(f"[TIMEOUT] Tag {expected_tag} expected student {expected_student} but no matching face.")
                # Log as absent/mismatch
                mark_attendance(expected_student, "MISMATCH", method="rfid_no_face")
            else:
                print(f"[TIMEOUT] Unknown tag {expected_tag} scanned but no face matched.")
            expected_student = None
            expected_tag = None
            expected_until = None

        
        # Draw a small overlay about expected tag (optional)
        # Show countdown timer while waiting for face after tag scan
        if expected_tag and expected_until:
            remaining = (expected_until - datetime.now()).total_seconds()
            if remaining > 0:
                timer_text = f"Show your face! Time left: {int(remaining)}s"
            else:
                timer_text = "Time expired!"
        # Display at top-left corner
            info = f"Tag waiting: {expected_tag}"
            cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, timer_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped. Attendance saved to DB and CSV.")

def export_attendance_csv(out="attendance_export.csv"):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    conn.close()
    df.to_csv(out, index=False)
    print(f"Exported attendance to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RFID+Face Attendance System")
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--enroll", action="store_true", help="Enroll a new student (capture images + add to DB)")
    sub.add_argument("--train", action="store_true", help="Train the face recognizer from dataset")
    sub.add_argument("--recognize", action="store_true", help="Run recognition and mark attendance")
    sub.add_argument("--export", action="store_true", help="Export attendance DB to CSV")
    parser.add_argument("--serial", type=str, default=None, help="Serial port for RFID (e.g., COM3 or /dev/ttyUSB0)")
    args = parser.parse_args()

    init_db()
    if args.enroll:
        name = input("Student name: ").strip()
        reg_no = input("Registration number (unique): ").strip()
        use_tag = input("Scan tag now via serial (y) or enter tag manually (m) or skip (s)? [y/m/s]: ").strip().lower()
        tag = None
        if use_tag == 'y' and args.serial:
            print("Please scan tag on your RFID reader now...")
            # simple blocking read from serial for enrollment
            try:
                with serial.Serial(args.serial, SERIAL_BAUD, timeout=10) as ser:
                    start = time.time()
                    while time.time() - start < 15:
                        raw = ser.readline().decode(errors='ignore').strip()
                        if raw.startswith("TAG:"):
                            tag = raw.split("TAG:")[-1].strip()
                            print("Captured tag:", tag)
                            break
            except Exception as e:
                print("Serial error:", e)
        elif use_tag == 'm':
            tag = input("Enter RFID tag ID (hex): ").strip()
        enroll_student(name, reg_no, rfid_tag=tag, serial_port=args.serial if use_tag=='y' else None)
    elif args.train:
        train_model()
    elif args.recognize:
        recognize_loop(serial_port=args.serial)
    elif args.export:
        export_attendance_csv()
