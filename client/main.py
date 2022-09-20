import socketio
import time

# standard Python
sio = socketio.Client()


@sio.event
def connect():
    print("I'm connected!")
    sio.emit("chat", {"x": 12, "y": 42, "car_id": 2, "sin": 0.2, "is_car": True})


@sio.event
def connect_error():
    time.sleep(5000)
    print("The connection failed!")
    sio.connect("https://7153-134-17-160-209.eu.ngrok.io/")


@sio.event
def message(data):
    print("I received a message!")


@sio.on("chat")
def on_message(data):
    print("Price Data ", data)


sio.connect("https://7153-134-17-160-209.eu.ngrok.io/")


@sio.event
def disconnect():
    # perform some user management stuff
    # perform some cleaning as well
    print("disconnect")
    try:
        sio.disconnect()
        sio.connect("https://7153-134-17-160-209.eu.ngrok.io/")
    except:
        print("zxc")
        time.sleep(5)
        sio.connect("https://7153-134-17-160-209.eu.ngrok.io/")
