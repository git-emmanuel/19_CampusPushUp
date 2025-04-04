
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
# poetry lock # to regenerate in case the .toml file is changed by hand
poetry show --outdated
# poetry install # to install the dependencies at the end
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
docker images && echo '=====' &&  docker ps && echo '====='
docker build -t acv_fitness_project .
# docker run --rm -p 8000:8000 -e SDL_AUDIODRIVER=dummy acv_fitness_project:latest # this would have been too easy ....
xhost +local:docker
docker run --rm -p 8000:8000 -e SDL_AUDIODRIVER=dummy --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix acv_fitness_project:latest
```
<!-- libEGL warning: DRI3: Screen seems not DRI3 capable
libEGL warning: DRI2: failed to authenticate
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1743768749.454935       1 gl_context_egl.cc:85] Successfully initialized EGL. Major : 1 Minor: 5
I0000 00:00:1743768749.461602      71 gl_context.cc:369] GL version: 3.2 (OpenGL ES 3.2 Mesa 22.3.6), renderer: llvmpipe (LLVM 15.0.6, 256 bits)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1743768749.534921      47 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1743768749.582529      50 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1743768751.468965      53 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
pygame 2.6.1 (SDL 2.28.4, Python 3.12.9)
Hello from the pygame community. https://www.pygame.org/contribute.html -->



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

Add Docker's Official Repository
```bash
sudo apt update
# ...
```

Install Docker Engine
`sudo apt install docker-ce`

Install Docker Desktop
`sudo apt install ./Downloads/docker-desktop-amd64.de`

Error message indicating that the path /tmp/.X11-unix is not shared from the host and is not known to Docker. 
- Go to Preferences (or Settings on Windows) in Docker Desktop.
- Navigate to the Resources section.
- Under File Sharing, click on the + icon to add a new directory.
- Add the /tmp/.X11-unix directory to the list of shared paths.

`docker run --rm -p 8000:8000 -e SDL_AUDIODRIVER=dummy --privileged --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix emmanuelxdocker/acv_fitness_project:latest`


