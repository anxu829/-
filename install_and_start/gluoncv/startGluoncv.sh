sudo docker stop gluoncv_xuan
sudo docker rm gluoncv_xuan
sudo nvidia-docker run --name gluoncv_xuan -it \
  -v /home/xuan/:/xuan \
  -p $1:$1 anxu5829/detectron:c2-cuda9-cudnn7 /bin/bash -c "cd / && jupyter lab --port $1 --ip 0.0.0.0 --allow-root"

