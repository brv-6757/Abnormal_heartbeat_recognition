import paho.mqtt.client as mqtt
import time
ecg_data = []
import numpy as np
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import json
from modelcode import AutoEncoder,intrpl

aws = AWSIoTMQTTClient("myClientID") 

aws.configureEndpoint("a2f1ivdkg6honi-ats.iot.ap-south-1.amazonaws.com", 8883)  
aws.configureCredentials("/home/rohit/Desktop/aws/AmazonRootCA1 (2).pem", 
                            "/home/rohit/Desktop/aws/7abde165d62697fbf84c7340890600ceb8e6275c8331e07c38db3096457bc4ee-private.pem.key", 
                            "/home/rohit/Desktop/aws/7abde165d62697fbf84c7340890600ceb8e6275c8331e07c38db3096457bc4ee-certificate.pem.crt")

aws.configureOfflinePublishQueueing(-1) 
aws.configureDrainingFrequency(2)  
aws.configureConnectDisconnectTimeout(10)  
aws.configureMQTTOperationTimeout(5)  
aws.connect()

def publish_data(anomaly_flag):
    topic = "health/metrics"  
    current_time = time.time()
	
    if anomaly_flag:
        message = {
            "anomaly": "heart_disease",
            "description": "Anomalous ECG pattern detected",
            "timestamp": int(current_time)
        }
        message_json = json.dumps(message)
        try:
            aws.publish(topic, message_json, 1)  
            print(f"Published: {message_json}")
        except Exception as e:
            print(f"Failed to publish message: {e}")

ae = AutoEncoder()
def on_connect(client,userdata,flags,rc):
	global flag_connected
	flag_connected = 1
	client.subscribe("ecg/data")
	print("connected")
	
def on_disconnect(client,userdata,flags,rc):
	global flag_connected
	flag_connected = 0
	print("disconnected")
	
def esp32_msg(client,userdata,msg):
	global ecg_data
	# global anamoly_flag
	# anamoly_flag = 0
	#print("ecg sensor data:",end="")
	data = msg.payload.decode("utf-8")
	ecg_data = np.array(list(map(float,data.split(','))))
	#print(ecg_data.shape)
	
	result = ae.call(intrpl(ecg_data.reshape(1,50)))
	print("Data received")
	if(result > ae.threshold) :
		print(f"Abnormal ECG Detected with model error: {result}")
		publish_data(1)
	else:
		print(f"Normal ECG with model error: {result}")

client = mqtt.Client("rpi_client1")#,callback_api_version=5)
flag_connected = 0
# anamoly_flag = 0
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = esp32_msg
client.connect("localhost",1883)
client.subscribe("ecg/data")
client.loop_start()
# client.message_callback_add("rpi/broadcast",callback_rpi_broadcast)
#while True:
	#time.sleep(4)

start_time = time.time()

while True:
    elapsed_time = int(time.time() - start_time)
    if(flag_connected==0):
       print("Trying to connect to esp32 Sensor")
    time.sleep(5)
        
