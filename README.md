# Parallel Computing Challenge 1: Parallel Merge Sort

**Team:** Ettore Cirillo, Angelo Notarnicola, Stefano Pedretti

## 📌 Overview
This project was developed for the Parallel Computing course (A.Y. 2024/2025). The goal is to implement the Merge Sort algorithm using OpenMP to exploit the parallel nature of the divide-and-conquer approach. By parallelizing the sorting process, the challenge aims to reduce execution time and optimize performance in multi-core environments.

## 🛠️ Technologies
* **Language:** C++.
* **Libraries:** OpenMP.
* **Compiler & Hardware:** Compiled using GCC with OpenMP support and tested on a 12-core MacBook Pro with an M3 Pro processor.
* **Core Concepts:** Divide-and-conquer algorithm, thread management, and empirical performance tuning.

## 🚀 Key Features
* **Parallel Execution:** Generates multiple threads to reorder small portions of an array simultaneously, before merging them to obtain the final ordered array.
* **Dynamic Sequential Switch:** Incorporates a tree-depth limit and a specific cut-off parameter to switch from parallel to sequential execution. This avoids the overhead caused by creating and managing threads on very small arrays.
* **Performance Scaling:** Demonstrated significant performance improvements, achieving a speedup of up to 6.155x for arrays of 10,000,000 elements compared to the serial approach.
