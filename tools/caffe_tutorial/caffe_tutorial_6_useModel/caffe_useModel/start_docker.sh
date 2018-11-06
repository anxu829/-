container='detectron_tutorial'
image='anxu5829/caffe:caffe_withopencv'
sudo docker stop $container 
sudo docker rm $container
sudo docker run --name $container   -it \
           -v /data2/xuan/mtcnn:/xuan \
	   -p $1:$1 $image \
	   /bin/bash \
	   -c "cd / && jupyter lab --port $1  --ip 0.0.0.0 --allow-root"
	    

