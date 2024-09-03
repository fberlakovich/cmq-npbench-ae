This is a fork of the [NPBench](https://github.com/spcl/npbench) repository to evaluate Cross-Module Quickening (see our paper [Cross Module Quickening - The Curious Case of C Extensions](https://ucsrl.de/publications/cmq-ecoop24-preprint.pdf)).
In addition to the baseline CPython and NumPy implementations, building and benchmarking requires the [Modified CPython](https://github.com/fberlakovich/cmq-ae) and [Modified NumPy](https://github.com/fberlakovich/cmq-numpy-ae).
The [Dockerfile](Dockerfile) builds the docker file we submitted to the artifact evaluation.
Due to a yet unresolved bug with the docker image (see #1), running the evaluation with the image does not complete in a reasonable time.

The most important changes compared to the original NPBench are:
* add Phoronix benchmarks
* add CMQ as a framework
* add the ability to collect various statistics (requires cooperation of the respective Python runtime)
* add the ability to generate tables and figures from the benchmark results

You can use the [quickstart.py](quickstart.py) script to run the benchmarks.
However, the script requires the differen Python environments (baseline and CMQ) to be setup already.
An easier way is to build and run the Docker image, once the bug is resolved.