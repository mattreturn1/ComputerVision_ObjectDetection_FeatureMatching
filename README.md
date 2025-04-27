# Installation & Execution Pipeline

Follow these steps to set up and run the project:

---

## Table of Contents

- [1. Download and Extract](#1-download-and-extract)
- [2. Open a Terminal in the Project Folder](#2-open-a-terminal-in-the-project-folder)
- [3. Launch the OpenCV Singularity Environment](#3-launch-the-opencv-singularity-environment)
- [4. Build the Project with CMake](#4-build-the-project-with-cmake)
  - [Generate the Makefile](#generate-the-makefile)
  - [Compile the Project](#compile-the-project)
- [5. Run the Executable](#5-run-the-executable)

---

## 1. Download and Extract

- Download the ZIP file.
- Extract it to your desired folder.

---

## 2. Open a Terminal in the Project Folder

- **GUI method:**  
  Right-click inside the extracted folder and select **"Open in Terminal"**.

- **Terminal command:**

  ```bash
  cd /path/to/yourrepo-main
  ```

  Make sure you're in the folder where you extracted the project before running any commands.

---

## 3. Launch the OpenCV Singularity Environment

- On the VM, run the following command to enter the Singularity container:

  ```bash
  start_opencv
  ```

  **Note:** If this is your first time using the container, you might need to configure the Singularity container before running this command. Please refer to the setup documentation for the container configuration if necessary.

---

## 4. Build the Project with CMake

### Generate the Makefile

-
  ```bash
  cmake .
  ```

This step generates the **Makefile**, which contains the rules for building your project.

### Compile the Project

-
  ```bash
  make
  ```

This command compiles the project using the generated Makefile. If you encounter errors during this step, ensure you have the necessary dependencies and that the environment is correctly set up.

---

## 5. Run the Executable

- Execute the compiled program:

  ```bash
  ./IntermediateProjectCV
  ```

  **Note:** The executable will be located in the build folder if you used an out-of-source build, or directly in the project folder if you did an in-source build.

---