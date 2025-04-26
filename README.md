## Installation & Execution Pipeline
Follow these steps to set up and run the project:

### 1. Download and Extract

- Download the zip file and extracts it in the desired folder

### 2. Open a Terminal in the Project Folder

- If using a GUI, right-click inside the folder and select "Open in Terminal".

### Alternatively, use the terminal command:

- cd /path/to/yourrepo-main

### 3. Launch the OpenCV Singularity Environment on the VM

- Run the following command to enter the Singularity container:

  start_opencv

### 4. Build the Project with CMake

#### Generate the Makefile

- cmake .

#### Compile the project

- make

### 5. Run the Executable

- ./IntermediateProjectCV