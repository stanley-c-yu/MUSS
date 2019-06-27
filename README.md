__Under development__

# Multi- vs. single-site accelerometer sensing for posture and activity recognition using machine learning (MuSS)

_source codes and data_

## Citation

In-submission.

## Dependencies

1. Python 3.6.5 or above
2. `pipenv` package. Install using `pip install pipenv`
3. `git` (optional)
4. Recommend at least 8-core workstation, otherwise computation will be slow
5. [`graphviz`](https://www.graphviz.org/download/) (optional, required to generate workflow diagram pdf)

## Replication of results

```bash
>> pipenv run reproduce --help
```

Run above command at the root of the repository to see the usage of the reproduction script.

The reproduct script will do the following,

1. Download and unzip the raw dataset
2. Compute features and convert annotations to class labels
3. Run LOSO cross validations on datasets of different combinations of sensor placements
4. Compute metrics from the outputs of the LOSO cross validations
5. Generate publication tables and graphs 

### Example

Run with multi-core processing and memory and time profiling on a new session folder. Overwrite data or results if they are found to exist.

```bash
>> pipenv run reproduce --parallel --profiling --run-ts=new --force-fresh-data=True
```

You may find intermediate and publication results in `./muss_data/DerivedCrossParticipants` in a folder prefixed with `product_run`.

### Sample reproduction results

Check [here](https://github.com/qutang/MUSS/releases/latest/download/sample_reproduction_results.tar.gz) for a sample reproduction results.
