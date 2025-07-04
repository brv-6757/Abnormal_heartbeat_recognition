import paho.mqtt.client as mqtt
import time
ecg_data = []
results = []
i=0
import numpy as np
from modelcode import AutoEncoder,intrpl

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
	global i
	print("ecg sensor data:",end="")
	data = msg.payload.decode("utf-8")
	data = np.array(list(map(float,data.split(','))))
	res = ae.call(intrpl(data.reshape(1,50)))
	print(res)
	#print(intrpl(ecg_data.reshape(1,100))[0]-ecg_data)
	#ecg_data.append(data)
	#results.append(res)
	
		

client = mqtt.Client("rpi_client1")
flag_connected = 0
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = esp32_msg
#client.message_callback_add("rpi/broadcast",callback_rpi_broadcast)
client.connect("localhost",1883)




client.subscribe("ecg/data")
client.loop_start()
print("setup complete")
while True:
	time.sleep(4)
	if(flag_connected!=1):
		print("trying to connect to mqtt server....")




  
























