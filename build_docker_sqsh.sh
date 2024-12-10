#!/bin/bash

help() {
	# Display Help
	echo "runs docker container script to output in 'sqsh' format."
	echo "Syntax: script [-c|-d]"
	echo "options:"
	echo "d location of the docker file."
	echo "c container name."
	echo "RUN IN SUDO IF DOCKER IS UNDER SUDO"
	echo
}

image_name=""
docker_file=""
setc=0
setd=0

while getopts ":c:d:" option; do
	case $option in
	c)
		image_name=$OPTARG
		setc=1
		;;
	d)
		docker_file=$OPTARG
		setd=1
		;;
	\?) # Invalid option
		echo "Error: Invalid option"
		help
		exit
		;;
	esac
done

if [[ $setc == 1 ]] && [[ $setd == 1 ]]; then
	exec_docker="docker build -f $docker_file . -t $image_name --no-cache"
	echo $exec_docker
	eval " $exec_docker"
	exec_enroot="enroot import dockerd://$image_name"
	echo $exec_enroot
	eval " $exec_enroot"
else
	help
fi
