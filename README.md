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
However, the script requires the different Python environments (baseline and CMQ) to be setup already.

The docker image aims to reproduce the results given in the paper, in particular:
* runtime performance as discussed in Section 7.3 (results may vary depending on machine)
* cache statistics as discussed in Section 7.4

### Runtime evaluation
To perform a full evaluation run: `docker run cmq-ae benchmark`.
This command runs the benchmarks with the same parameters as in the paper and prints a table with the results at the end.

On our machines, the full evaluation run took ~3-4 hours.
If you want shorter runtimes, you can either use a smaller preset (input size), fewer repetitions or both:
* smaller input size (preset `M`): `docker run cmq-ae --preset M -r 20 -b all -f cmlq -f numpy --cmlq-path=venv_cmq/bin/python --output table`
* fewer repetitions (`5` repetitions): `docker run cmq-ae --preset paper -r 5 -b all -f cmlq -f numpy --cmlq-path=venv_cmq/bin/python --output table`
* both (preset `M` with `5` repetitions): `docker run cmq-ae --preset M -r 5 -b all -f cmlq -f numpy --cmlq-path=venv_cmq/bin/python --output table`

### Statistics
To collect statistics, run: `docker run cmq-ae statistics`
This command runs a special CMQ build that collects various statistics and prints a table with the aggregated results from the paper.
On our machines the statistics collection took 1-2 hours.
If you want shorter runtimes, use a smaller preset and use fewer repetitions:
`docker run cmq-ae --preset M -r 10 -b all -f cmlq -f numpy --cmlq-path=venv_cmq_stats/bin/python --output stats`

### Results
The results are stored in a sqlite database located in `/output` in the container.
If you want to inspect the data, use a persistent docker volume or mount a host directory to `/output`.
For example: `docker run -v $PWD/localdir:/output cmq-ae statistics`
By specifying `--output figure` in the benchmark command, you can generate the figure from the paper:
```
docker run -v $PWD/localdir:/output cmq-ae benchmark --output figure
ls -l localdir/performance.png
```