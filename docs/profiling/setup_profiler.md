
# Profiling

Profiling is important in kernel development.


## View your kernel trace


### Step 1: Install the profiler on your AMD machine.

```bash
mkdir -p rocprofiler-setup
cd rocprofiler-setup

git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
sudo apt-get update
sudo apt-get install libdw-dev libelf-dev elfutils
cmake -B rocprofiler-sdk-build -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm/ rocprofiler-sdk-source
cmake --build rocprofiler-sdk-build --target all --parallel 32
cmake --build rocprofiler-sdk-build --target install

# if using ubuntu 22.04 (like the rocm7.0 preview container)
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.1/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb
sudo dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb

# if using ubuntu 24.04 (like the rocm 6.4.1 container we used on the mi325x)
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.1/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb
sudo dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb

git clone https://github.com/ROCm/aqlprofile.git
cd aqlprofile
./build.sh
cd build
sudo make install
cd ../../..
rm -rf rocprofiler-setup
```

### Step 2: Collect the trace.

Use these commands to collect traces for your kernel. This will produce a new trace directory ```my_dir``` and dump the trace outputs there. This assumes that your kernel is being run within a file called ```test_python.py```, and the profiler will produce a separate trace for every kernel called within the file. If you want to profile a C++ binary, for instance called ```matmul``` then you can replace ```-- python3 test_python.py``` with ```-- ./matmul```. 

```bash
rocprofv3 --att=true \
          --att-library-path /opt/rocm/lib \
          -d my_dir \
          -- python3 test_python.py
```

### Step 3: View the trace.

1. Inside your directory ```my_dir``` there will be many different folders (```ui_...```). Open each folder and view the ```code.json``` folder contained within it -- see if the ```code.json``` mentions your kernel's name (e.g., you might see the name of your HK kernel or AITER). Download the **entire** ```ui_...``` folder that contains the contents for your kernel of interest to your local computer. 
<div align="center" >
    <img src="./assets/download.png" height=250 alt="Step 1" style="margin-bottom:px"/> 
</div>

2. Next, install an application on your local computer to be able to visualize the trace:
Clone below and build from source on your local filesystem:
```bash
# clone
git clone https://github.com/ROCm/rocprof-compute-viewer
cd rocprof-compute-viewer/
# on your local mac computer (for other platforms: https://github.com/ROCm/rocprof-compute-viewer?tab=readme-ov-file#building-from-source)
brew install qt@6
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$(brew --prefix qt@6)
make -j
```

3. Then on your local computer open the installed ```rocprof-compute-viewer``` application and import the entire ```ui_...``` folder as shown:
<div align="center" >
    <img src="./assets/rocprof.png" height=250 alt="Step 1" style="margin-bottom:px"/> 
</div>

This is what it will look like when the folder is imported:
<div align="center" >
    <img src="./assets/final.png" height=250 alt="Step 1" style="margin-bottom:px"/> 
</div>



### Utilities.

- Additional details on how to use ```rocprof-compute-viewer``` once it's set up and your trace is loaded into it can be found here: https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/latest/how-to/using_compute_viewer.html#using-compute-viewer. 
- You can additionally run ```extract_asm_from_rocprof_json.py code.json kernel.s``` to take the assembly contained in a ```code.json``` that you are interested in, and cleanly extract its contents into a file called ```kernel.s```. We provide this python script in this folder. 


### Notes.

You might encounter a few failure modes in the above steps:

1. No package 'libdw' found:
```bash
sudo apt-get update
sudo apt-get install libdw-dev libelf-dev elfutils
```

2. View decoder errors:
```bash
LD_DEBUG=libs rocprofv3 \
  --att=true \
  --att-library-path /opt/rocm/lib \
  -d my_dir \
  -- python3 test_python.py 2>&1 | tee decoder_debug.log
```

Check detailed information about what is going wrong.


## Performance counters

In addition to visualizing traces, sometimes we want to emasure various performance statistics (cache hit rates, number of waves per kernel, proportions of different instruction types, bank conflicts, etc.) for our kernel. 

We provide a bash script to help you collect counters. 
```bash
# 1. Open the script and make sure the application name (python3 test_python.py) is correct and calls your kernels. 

# 2. Run each command in:
bash profile_pmc_counters.sh
```

We provide the python script to help you view the counter outputs nicely.
```bash
# 1. Open the file and make sure that KERNEL_MAP contains the correct kernels (view the counter outputs if you don't know your kernel names)

# 2. Run.
python analyze_pmc_counters.py
```




