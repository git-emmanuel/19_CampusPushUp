
# Computer Vision Project

<img src="media/push-up_logo.png" width="200" />

Welcome to the CampusPushUp repository. 

This project is a collaborative effort by Romain, Emmanuel, and Karim.


# Running the project

In the terminal, run: ` python run.py 'media/yoga.mp4' `


# Setting up poetry

```bash
pip install poetry
poetry init # to create the toml file, then fill it manually with the packages needed
# conda list --export > requirements.txt  # Export the environment dependencies
# poetry add $(cat requirements.txt)
poetry add numpy
poetry add opencv-python
poetry add mediapipe
poetry add pygame 
poetry add joblib
poetry add pandas
# math, sys, time already installed
poetry add scikit-learn
poetry run python run.py # to check that everything is fine
```

```bash
"numpy (>=1.21.0,<3.0.0)",
"opencv-python (>=4.11.0.86,<5.0.0.0)",
"mediapipe (>=0.10.21,<0.11.0)",
"pygame (>=2.6.1,<3.0.0)",
"joblib (>=1.4.2,<2.0.0)",
"pandas (>=2.2.3,<3.0.0)",
"scikit-learn (==1.5.2)"
```

```bash
# docker images
# docker container ls -a
docker system prune
# poetry lock # to regenerate in case the .toml file is changed by hand
docker images && echo '=====' &&  docker ps && echo '====='
docker build -t acv_fitness_project .
# docker run --rm -p 8000:8000 -e SDL_AUDIODRIVER=dummy acv_fitness_project:latest # this would have been too easy ....
xhost +local:docker
docker run --rm -p 8000:8000 -e SDL_AUDIODRIVER=dummy --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix acv_fitness_project:latest
```

# Pushing to docker hub

```bash
docker login
docker tag acv_fitness_project:latest emmanuelxdocker/acv_fitness_project
docker images && echo '=====' &&  docker ps
docker push emmanuelxdocker/acv_fitness_project:latest
```

`du -sh $(pip show $(pip list --format=freeze | cut -d= -f1) | grep Location | awk '{print $2}')/* 2>/dev/null | sort -hr`

`du -ch $(pip show $(pip list --format=freeze | cut -d= -f1) | grep Location | awk '{print $2}')/* 2>/dev/null | grep total`


# Pulling from docker hub




