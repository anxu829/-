docker stop gluoncv_xuan
docker rm gluoncv_xuan
nvidia-docker run --name gluoncv_xuan -it \
  -v /home/xuan/:/xuan \
  -p $1:$1 anxu829/gluoncv:lab  /bin/bash -c "cd / && jupyter lab --port $1 --ip 0.0.0.0 --allow-root"

