

### Setup mojo

Docker and environment (https://docs.modular.com/mojo/manual/get-started/):
```
# This docker recommended by mojo gives errors (11/8/2025)
docker.io/modular/max-amd-base 
So we use beta:

podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    --entrypoint /bin/bash \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta 

# if you don't have it, install pixi
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="/root/.pixi/bin:$PATH"

# create a project
pixi init life \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd life

# install the modular conda package
pixi add modular

# setup the VM
pixi shell
```

Run: 
```
mojo kernel.mojo
```



