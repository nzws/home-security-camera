from dotenv import load_dotenv
load_dotenv()

from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt
from time import sleep
from datetime import datetime
import os
import requests
import json
import numpy as np

SHOW_WINDOW = os.getenv("SHOW_WINDOW") == "true"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX")) if os.getenv("CAMERA_INDEX") else 0
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST") if os.getenv("MQTT_BROKER_HOST") else "localhost"
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT")) if os.getenv("MQTT_BROKER_PORT") else 1883
ACTIVATE_MONITOR_TOPIC = os.getenv("ACTIVATE_MONITOR_TOPIC")
NOTIFY_TOPIC = os.getenv("NOTIFY_TOPIC")
NOTIFY_DISCORD_WEBHOOK = os.getenv("NOTIFY_DISCORD_WEBHOOK")

if ACTIVATE_MONITOR_TOPIC is None or NOTIFY_TOPIC is None:
    print("Not configured correctly!")
    exit()

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_INDEX)
monitor_activated = False
prev_alerted = False

def send_to_discord(text: str, files: dict):
    url = NOTIFY_DISCORD_WEBHOOK
    if url is None:
        return

    payload = {
        "content": text + " at " + datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    }

    try:
        requests.post(url, data={"payload_json": json.dumps(payload)}, files=files)
    except Exception as e:
        print("Failed to send to Discord", e)

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")

    client.subscribe(ACTIVATE_MONITOR_TOPIC)

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode("utf-8")
    print(f"Received message: {topic} {payload}")

    if topic == ACTIVATE_MONITOR_TOPIC:
        global monitor_activated
        monitor_activated = payload == "1"

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message

mqttc.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)

mqttc.loop_start()

def put_timestamp(frame):
    current_timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    cv2.putText(frame, current_timestamp, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

def video_writer(cap, fps, time, path = "saved/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".mp4"):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for _ in range(fps * time):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = put_timestamp(frame)
        out.write(frame)

    out.release()

    return path

while cap.isOpened():
    success, frame = cap.read()

    if success is False:
        continue

    key = cv2.waitKey(10)
    if key == 27:
        break

    if monitor_activated is False:
        continue

    results = model(frame, verbose=False)
    person_count = 0

    for result in results:
        boxes = result.boxes.cpu().numpy()

        for box in boxes:
            name = result.names[int(box.cls[0])]

            if name == "person":
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                person_count += 1

    frame = put_timestamp(frame)

    if person_count > 0:
        cv2.putText(frame, f"Person: {person_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    is_alert = person_count > 0
    if prev_alerted != is_alert:
        try:
            mqttc.publish(NOTIFY_TOPIC, "1" if is_alert else "0")
        except Exception as e:
            print("Failed to publish to MQTT", e)

        prev_alerted = is_alert

        if is_alert:
            _, image = cv2.imencode(".jpg", frame)
            send_to_discord("Detect person", {
                "file": ("image.jpg", image.tobytes(), "image/jpeg")
            })
            open("saved/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg", "wb").write(image.tobytes())

            fps = 30
            time = 30
            video_path = video_writer(cap, fps, time)
            print("Video saved at", video_path)
            send_to_discord("Detect person", {
                "file": ("video.mp4", open(video_path, "rb"), "video/mp4")
            })

            sleep(60)

    if SHOW_WINDOW:
        cv2.imshow("frame", frame)

    sleep(5)

mqttc.loop_stop()
cap.release()

if SHOW_WINDOW:
    cv2.destroyAllWindows()
