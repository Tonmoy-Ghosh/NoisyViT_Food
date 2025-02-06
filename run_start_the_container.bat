::Shell script to run the docker container with worker tasks

::Build and run the container
@echo [101;93m noisy_nn WORKER [0m
set workername=noisy_nn

::Delete previous container if exists
docker rm -f %workername%

::copy ..\lib_jitai\jitai_config.ini .
::mkdir lib_jitai
::copy ..\lib_jitai\*.py .\lib_jitai

::Build and run image:
docker build  --no-cache -t %workername% .

::docker run --cpus=2 --gpus all -it --shm-size=30g --memory-swap "80g" --memory="40g" --name %workername% -d -v "%cd%":/workercode --network jitaiNetwork %workername%
docker run --cpus=2 --gpus all -e TZ=America/Chicago -it --shm-size=30g --memory-swap "80g" --memory="40g" --name %workername% -d -v "%cd%":/workercode  -v jitai-img-vol:/jitairootdir --network jitaiNetwork %workername% 
