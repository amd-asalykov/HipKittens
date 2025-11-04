
## Setup MI355X

We currently use the following recent docker. 
```
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta

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
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash
```

## Common issues

If all the files become root-owned, run this command to fix it (for user id 1003 and guest id 1003):
```bash
# Show your user ID
id -u
# Show your group ID
id -g

# run the command below with your user and group ID
sudo chown -R 1003:1003 /workdir/
```


