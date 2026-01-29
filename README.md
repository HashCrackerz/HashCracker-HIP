<p align="left">
  <img width="150" alt="logo" src="https://github.com/user-attachments/assets/73297594-3581-4afd-ad0b-39d4bc0e66bf" />
</p>

# HashCracker (HIP)
## _Parallel SHA-256 Brute Force & Dictionary (salted) Password Cracker_

[🇬🇧 English](README.md) | [🇮🇹 Italiano](README-IT.md)

![alt text](https://img.shields.io/badge/Language-C++%20%7C%20CUDA-green)

![alt text](https://img.shields.io/badge/Platform-AMD%20(HIP)-red) 

![alt text](https://img.shields.io/badge/Algorithm-SHA256-purple)

Project for the Accelerated Computing Systems course at the Master's degree in Computer Engineering, University of Bologna.
**Parallel** application for password cracking through Brute Force attack on SHA-256 hashes (including salted) and dictionary attack, 
with performance comparison between Sequential (CPU) and Parallel (HIP for AMD GPU) implementations.

## 📝 Description
The project implements a **password cracker** that supports different attack modes to reverse SHA-256 hashes. 
The main goal is to demonstrate the speedup achievable by moving from serial execution on CPU to 
massively parallel execution on AMD GPU in this case.

## ⚙️ Features
- **Incremental Brute Force**: Dynamic password generation given a charset and length range (min-max).
- **Dictionary Attack**: Support for external wordlists.
- **Salt Support**: Handling of salted hashes (Brute Force and dictionary attack).
- **Multi-Platform**: Native CUDA code for **NVIDIA** and semi-automatic porting script for **AMD** HIP.

## 📂 Project Structure
- `Sequenziale/`: **Sequential** reference implementation (uses [OpenSSL](https://openssl-library.org/)).
- `HIP_NAIVE/`: **First GPU implementation** (global memory usage).
- `HIPv1/`: Memory optimization (Constant Memory usage for charset and target).
- `HIPv2/`: Kernel optimization (loop unrolling, register optimization for SHA-256).
- `UTILS/`: Support functions (file I/O, argument parsing).
- `SHA256_HIP/`: **HIP implementation for SHA256**, based on [mochimodev](https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu)'s implementation.
- `SHA256_HIP_OPT/`: **Optimized HIP implementation for SHA256** (used by HIPv2 and extension).
- `ESTENSIONE/`: contains the implementation of the project extension, i.e., dictionary attack and hash cracking with salt.
- `convert_to_hip.ps1`: PowerShell script for **semi-automatic porting** to AMD ROCm/HIP (and guide for remaining porting operations).
- `kernel_[project_version].cu`: file to run the corresponding version.
All HIP[project_version] versions (executed by their respective kernel files) depend on 
`UTILS` and `SHA256_HIP` files, except for v2 which uses `SHA256_HIP_OPT` (optimized version) instead of `SHA256_HIP`.
Note that the HIP implementation is substantially analogous to the CUDA one reported in the [HashCracker-CUDA](https://github.com/HashCrackerz/HashCracker-CUDA) repository.

The porting script handles appropriate translation of functions, types and importing or removing libraries, as well as
converting imports with double quotes to angle brackets. Files are overwritten and renamed, but the
project structure remains the same. The files in this repository are the result of this conversion and 
refined manual porting (adding extern C to kernels, fixing thread invocation spacing and little else). 

## 🛠️ Requirements
- Hardware:
  - **AMD GPU**

- Software:
  - **OpenSSL** (for CPU implementation)
  - **C++ Compiler** (MSVC on Windows, GCC/Clang on Linux)
  - **PowerShell** (for porting script)
  - **ROCm** (to compile with hipcc and for AMD environment).
 
## 🚀 Compilation

### AMD HIP (Porting)
The project includes a script to automatically convert CUDA code to HIP.
1. Run the conversion script:
   ```powershell
   .\convert_to_hip.ps1
   ```
2. Manual check
   1. All kernels must be launched with triple angle bracket syntax without spaces (example `kernel <<< block, thread >>> (args)`).
   2. Kernel functions must be `__global__` and `extern "C" {}` (both in .cu and .cuh).
   3. Check compilation errors.
2. Compilation with hipcc (Windows example):
   ```powershell
      hipcc -fgpu-rdc -O3 -std=c++14 --offload-arch=native `
        kernel_estensione.cu `
        ESTENSIONE/DIZIONARIO/hip_dizionario.cu `
        ESTENSIONE/SALT/hip_salt.cu `
        SHA256_HIP/sha256.cu `
        UTILS/hip_utils.cu `
        UTILS/utils.cpp `
        -I. -I./ESTENSIONE/DIZIONARIO -I./ESTENSIONE/SALT -I./SHA256_HIP -I./UTILS `
        -I"D:\OpenSSL-Win64\include" `
        -L"D:\OpenSSL-Win64\lib\VC\x64\MTd" `
        -l libcrypto.lib `
        -D_CRT_SECURE_NO_WARNINGS `
        -o estensione_amd.exe
   ```
   _(change file names and dependencies based on the version to compile)_

## 💻 Usage
The program accepts command line parameters for maximum flexibility:
```cmd
./brute_force_cuda [<blockSize>] <hash_target> <min_len> <max_len> <file_charset> [<salt> <dictionary-yes/no> <dictionary_file>]
```
The `blockSize` must always be passed only in parallel GPU scripts (both CUDA and HIP). \
The dictionary (flag and file path) and salt must be passed only in extension scripts. \
_Note_: in the extension version `max_len` includes the salt length. 

Example: \
Search for the password of the hash (corresponding to "qwerty") with length 6, using the standard charset:
```cmd
./brute_force_cuda 256 qwerty 1 6 ASSETS/CharSet.txt
```


## 📊 Performance Analysis
Tests were conducted on:
- desktop with Ryzen 9 9900X and RX 9070XT

### Technical Deep Dive: Analysis
The SHA-256 algorithm is heavily **Compute-Bound**. The v2 implementation heavily uses registers to maintain the hash 
state and avoid local/global memory latencies. Although the high number of registers (118) limits the number of active warps 
(low occupancy), the single thread execution speed increases drastically. In this scenario, maximizing IPC 
(Instructions Per Cycle) proved more effective than maximizing parallelism at the latency level (Occupancy).

Furthermore, the use of smaller block sizes (64/128 threads) led to better performance compared to the classic 256, 
thanks to better management of the Tail Effect (wave quantization) and lower scheduling overhead.

The extension implementation has essentially the same performance as v2 (since it uses practically the same code), 
with the addition that for dictionary attack, the time in case of hit is certainly lower than testing all combinations.

## 👥 Authors
- [Andrea Vitale](https://github.com/WHYUBM)
- [Matteo Fontolan](https://github.com/itsjustwhitee)

## 📜 License
This project is distributed under the AGPL license. See the `LICENSE` file for details.
