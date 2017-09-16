# Batch Matrix Transpose Benchmark

This benchmark evaluates the performance of batch matrix transposition in Tensorflow. 
This benchmark includes a hard coded version of util.h that calculates optimal kernel launch configurations that needs to be modified for different architecture.
The paramters we modified for each architecture tested are:

| Arch | max thread num | number of SMs | Reference  |
|------|----------------|---------------|------------|
| K40  | 1024           | 15            |            |
| P100 | 1024           | 56            | [reference](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf) |

# Experimental Setup and Results

Results are included in the color\_coded\_results.xlsx Excel file. We have explored a vast majority of practical use cases. Specifically we will refer to three problem size dimensions when describing our test cases: batch size, matrix height and matrix weight. When presenting our data, the first two dimensions are serialized. Below is an excerpt from our python code that defines the problem space we explored:

```
batch_num = [2**i for i in range(5, 13)]
matrix_height = range(96, 2048, 16)
matrix_width  = range(2, 16)
```

Based on the data we have collected, our implementation shows 12% average speedup on K40 machine whilst demonstrating a 39% speedup on the P100 machine. Results in the spreadsheet are color-coded as such:

Red (Loss):    where our implementation is 10% or more slower than baseline.

Yellow (Win): where our implementation is 10% or more faster than baseline.

Uncolored (On Par): where our implementation is less than 10% slower/faster than baseline.

# To evaluate
To compile:
Modify Makefile to make sure that you are using the latest compute capability, we used 3.5 for K40 and 6.0 for P100.

Run `make`

To benchmark:

Run `python3 perf.py`
