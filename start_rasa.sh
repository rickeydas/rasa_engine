#!/bin/bash
sudo apt-get -y install python3-pip
sudo pip3 install -r requirements.txt
pkill -f "uvicorn server:app"

nohup uvicorn server:app --host 0.0.0.0 --port 8000 &
