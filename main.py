#main.py
from flask import Flask, render_template, request, redirect, url_for, send_file , flash
import os
import sqlite3
import subprocess
import pandas as pd

# Import your existing functions
import hybrid_attendance as hybrid 

app = Flask(__name__)

# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/students")
def students():
    rows = hybrid.get_students()
    return render_template("students.html", students=rows)


@app.route("/view")
def view():
    return render_template("view.html")

@app.route("/attendance")
def attendance():
    conn = sqlite3.connect(hybrid.DB_FILE)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    conn.close()
    return render_template("attendance.html", records=df.to_dict(orient="records"))

@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        name = request.form["name"]
        reg_no = request.form["reg_no"]
        rfid_tag = request.form["rfid_tag"] or None
        hybrid.enroll_student(name, reg_no, rfid_tag=rfid_tag)
    
        return redirect(url_for("index"))
    return render_template("enroll.html")

@app.route("/train")
def train():
    hybrid.train_model()
    return "Training complete. <a href='/'>Back</a>"

@app.route("/recognize")
def recognize():
    # NOTE: This opens webcam & blocks until stopped
    hybrid.recognize_loop(serial_port=hybrid.SERIAL_PORT)
    return "Recognition stopped. <a href='/'>Back</a>"

@app.route("/export")
def export():
    out = "attendance_export.csv"
    hybrid.export_attendance_csv(out)
    return send_file(out, as_attachment=True)

# ---------- RUN ----------
if __name__ == "__main__":
    hybrid.init_db()
    app.run(debug=True)
