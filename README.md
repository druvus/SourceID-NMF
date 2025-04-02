# SourceID-NMF
[SourceID-NMF: Towards more accurate microbial source tracking via non-negative matrix factorization.](https://doi.org/10.1093/bioinformatics/btae227)


A major challenge in analyzing the compositional structure of microbiome data is identifying its potential origins. Here, we introduce SourceID-NMF, a tool for precise microbial source tracking. SourceID-NMF utilizes a non-negative matrix factorization (NMF) algorithm to trace the microbial sources contributing to a target sample.


<div style="text-align: center;">
<img src='image/NMF.png' width="372" height="186">
</div>


## Support
For support using SourceID-NMF, please email: zhuang82-c@my.cityu.edu.hk

## Required Dependencies
Detailed package information can be found in requirement.txt. The main environment configuration we need includes:
* Conda
* Python >=3.8.13
* numpy >=1.24.3
* pandas >=2.0.3
* tqdm >=4.66.1
* scipy >=1.10.1

We suggest installing SourceID-NMF's dependencies in a conda environment.

The command is: 
```
conda env create -n nmf python numpy pandas tqdm scipy
conda activate nmf
```

Clone the repository and install SourceID-NMF in the nmf environment.

```
git clone https://github.com/username/sourceid-nmf.git
cd sourceid-nmf
pip install .
```

## Usage

SourceID-NMF provides two main commands:
- `track`: Perform source tracking analysis
- `evaluate`: Evaluate tracking performance against true proportions

### Source Tracking Command

```
sourceid-nmf track -i ./data/nmf_data.txt -n ./data/name.txt -o ./estimated_proportions.txt
```

#### Basic Parameters

```
Options
-i, --input        Path to input count table (tab-separated)
-n, --name         Path to name file: Data labels for input data
-o, --output       Path to output file for estimated proportions
-t, --thread       Max workers for multiprocessing operation (default: 20)
-e, --iter         Maximum number of iterations for the NMF model (default: 2000)
-r, --rho          The penalty parameter (default: 1.0, 0 for auto-selection)
-a, --weight       The weighting matrix factor (default: 1)
-c, --threshold    The convergence threshold (default: 1e-06)
-m, --mode         Operation mode: "normal" or "cluster" (default: normal)
-f, --cutoff       The clustering threshold for JSD distance (default: 0.25)
-p, --perf         Path to output performance metrics (optional)
```

#### Advanced Optimization Parameters

```
Options
--use-active-set   Enable active-set method to accelerate sparse data processing
--no-active-set    Disable active-set method (overrides auto-detection)
--adaptive-rho     Enable adaptive rho parameter for faster convergence
--fixed-rho        Use fixed rho parameter (no adaptation)
--auto-optimize    Automatically detect and apply optimal settings (default)
```

#### Global Parameters

```
Options
-v, --verbose      Increase verbosity (can be used multiple times)
--log              Path to log file (optional)
--version          Show version number and exit
```

### Input Data Format

Suppose a dataset has 19 sources, represented by D1, D2,... ,D19, and 9 sinks, represented by D20, D21,... , D28.

The input to SourceID-NMF is composed of two txt files:

`-i | --input`

The input count table containing sources and sinks (M by N). where M is the number of samples and N is the number of taxa. Row names are the taxa ids. Column names are the sample ids. Every column contains read counts for each sample.

The specific input table case is shown below:

| | D1 | D2 | D3 | ... | D19 | D20 |
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| taxon_1  |  0 | 5 | 0 | ... | 20 | 5 |
| taxon_2  |  20 | 5 | 0 | ... | 0 | 11 |
| taxon_3  |  0 | 13 | 210 | ... | 0 | 20 |
| taxon_4  |  80 | 6 | 0 | ... | 0 | 0 |
| taxon_5  |  4 | 38 | 0 | ... | 14 | 0 |
| ... | ... | ... | ... | ... | ... | ... |
| taxon_n  |  24 | 25 | 0 | ... | 0 | 14 |

`-n | --name`

The name table contains three columns, 'SampleID', 'Env' and 'SourceSink'. The 'SampleID' column describes the labels for each source data or sink data. The 'Env' column describes the environment to which each source or sink belongs, e.g. the first row Env = 'Electronics' means that the source was collected from Electronics. This 'SourceSink' column describes the source or sink to which the data belongs. 

The specific name table case is shown below:

| SampleID | Env |SourceSink |
| ------------- | ------------- |------------- |
| D1 | Electronic | Source |
| D2 | Hand | Source |
| D3 | Incubator | Source|
| D4 | Surface | Source|
| ... | ... | ... |
| D19 | Tubes | Source |
| D20 | fecal | Sink |

### Output Data Format

`-o | --output`

The output table contains all the estimated proportions (K by S+1), where K is the number of sinks and S is the number of sources (including an unknown source). The specific value in this table represents the contribution of each source to each sink. The sum of the proportions in each row is 1.

The specific output table case is shown below:

| | D1 | D2 | D3 | ... | D19 | Unknown |
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| D20 | 0.021288945 |	0.013085965 |	0.008938594 | ... |	0.005083895 |	0.485646292 |

### Performance Evaluation Command

To evaluate the performance of source tracking against known true proportions:

```
sourceid-nmf evaluate -e ./estimated_proportions.txt -t ./data/true_proportions.txt -o ./performance_results.txt
```

Parameters:

```
Options
-e, --estimated    Path to estimated proportions file
-t, --true         Path to true proportions file
-o, --output       Path to output performance results (default: proportion_perf.txt)
```

## Advanced Features

### Automatic Optimization

The current version of SourceID-NMF includes several advanced optimization features that can be automatically selected based on your data characteristics:

1. **Active-Set Method**: Accelerates processing of sparse data matrices by focusing computation on non-zero elements.

2. **Adaptive Rho Parameter**: Dynamically adjusts the penalty parameter during optimization for faster convergence.

3. **Parallel Processing**: Optimizes thread allocation between sink processing and ADMM optimization based on your data and available CPU cores.

4. **Data Characteristic Detection**: Automatically analyzes your data to select optimal parameters.

To use these features, simply add the `--auto-optimize` flag:

```
sourceid-nmf track -i ./data/nmf_data.txt -n ./data/name.txt -o ./estimated_proportions.txt --auto-optimize
```

### Source Clustering

For datasets with many similar sources, SourceID-NMF can cluster sources before analysis to improve performance:

```
sourceid-nmf track -i ./data/nmf_data.txt -n ./data/name.txt -o ./estimated_proportions.txt -m cluster -f 0.25
```

The `-f` parameter controls the Jensen-Shannon divergence threshold for clustering (lower values create more clusters).

## Demo

To run the demo data with default settings:

```
sourceid-nmf track -i ./data/nmf_data.txt -n ./data/name.txt -o ./estimated_proportions.txt
```

With advanced optimization:

```
sourceid-nmf track -i ./data/nmf_data.txt -n ./data/name.txt -o ./estimated_proportions.txt --auto-optimize
```

To evaluate performance on simulated data:

```
sourceid-nmf evaluate -e ./estimated_proportions.txt -t ./data/true_proportions.txt
```

## Simulation Data

Our simulated data was generated using the microbial data from the Earth's microbiome project [1]. It can be downloaded from http://ftp.microbio.me/emp/release1/otu_tables/closed_ref_greengenes/ [1]. We used the emp cr_gg_13 8.subset 2k.rare 10000.biom file from this link to simulate data.

## References

[1] Thompson, L. R. et al. A communal catalogue reveals Earth's multiscale microbial diversity. Nature 551, 457–463 (2017).
