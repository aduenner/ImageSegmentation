# Setting up the Docker Image 
(adapted from [Keras Docker Image](https://github.com/keras-team/keras/tree/master/docker))


## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

For GPU support install NVIDIA drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). 



## Building the container and pushing to docker

    $ make bash GPU=0
    $ exit
    $ docker image -ls
    $ docker tag <latest_image> <dockeruser>/<docker_repo>:latest
    $ docker push <docker_user>/<docker_repo>:latest
  
* <docker_user> is your docker user name
* <docker_repo> is the name of your docker repo
* <latest_image> is the image file associated with the most recent build (found with 'docker image ls')

## Running in Paperspace

    $ paperspace jobs create --container <docker_user>/<docker_repo> --machineType P4000 --command '<Your Command Here>' --ports <port1>:<port2> --project Job Builder --workspace git+<git_repo>
Example:

    $ paperspace jobs create --container duenner/paperspace-keras --machineType P4000 --command jupyter notebook --ip=0.0.0.0 --ports 8888:8888 --project Job Builder --workspace git+https://github.com/aduenner/ImageSegmentation.git
