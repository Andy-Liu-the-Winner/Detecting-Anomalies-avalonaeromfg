# Detecting Anomalies in Manufacturing Data

## **Setup Instructions**

### **1. Download CSV Files**
Download and store the CSV files as shown below:

![Screenshot 2025-03-13 at 04 18 10](https://github.com/user-attachments/assets/0180ba03-7e0a-49c3-a9b0-898e50238187)

### **2. Organize the CSV Files**
- Rename each CSV file to its corresponding **cycle number**.
- Store the files in one of the following directories:
  - `abnormal_csv/` for anomaly data
  - `normal_csv/` for normal data

### **3. Run the Detection Script**

To execute the anomaly detection, run:
```bash
python train.py
```

### **4. Results**
After running the script, you will see the detected anomalies along with a visualization of tool degradation over cycles.

![Screenshot 2025-03-13 at 04 22 38](https://github.com/user-attachments/assets/d5c7f55b-4f7f-42d3-950e-94b19a8f97ac)
![Screenshot 2025-03-13 at 04 22 59](https://github.com/user-attachments/assets/0e4d852a-aa0a-4be3-953b-5ed9037ffb36)
![Screenshot 2025-03-13 at 04 23 18](https://github.com/user-attachments/assets/48207354-3a1f-4e89-91b2-420e6c997d28)
