# batch_matrix_transpose_benchmark

This benchmark evaluates the performance of batch matrix transposition in Tensorflow. 
This benchmark includes a hard coded version of util.h that calculates optimal kernel launch configurations that needs to be modified for different architecture.
The paramters we modified for each architecture tested are:

| Arch | max thread num | number of SMs | Reference  |
|------|----------------|---------------|------------|
| K40  | 1024           | 15            |            |
| P100 | 1024           | 56            | [reference](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf) |


To compile:
Modify Makefile to make sure that you are using the latest compute capability, we used 3.5 for K40 and 6.0 for P100.

Run `make`

To benchmark:

Run `python3 perf.py`
