## Histogram of array of numbers using Thrust

Thrust is a Standard Template Library for CUDA that contains a Collection of data parallel primitives (eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance CUDA code.

This folder contains the implementation of histogram of array of integers using Thrust.

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out <input_filename> <output_filename> 
```
