#!/bin/bash

NAME=akraradet-lab
SSH_PORT=21413

cd "`dirname $0`"
cd image
docker build . -t $NAME
cd ..
docker run -d -p ${SSH_PORT}:22 -v $HOME/RTML:/root/RTML --gpus all --name $NAME $NAME
# docker run -d -p 21413:22 -v $HOME/RTML:/root/RTML --gpus all --name akraradet-lab akraradet-lab
