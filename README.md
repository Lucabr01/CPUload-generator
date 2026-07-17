# CPUload Generator
An infinite CPU load trace generator for EnergyPlus environments, designed for integration with Sinergym. It introduces stochastic electrical loads based on statistics extracted from real CPU data, overcoming the deterministic load profiles commonly used in HVAC control simulations. The generated traces preserve realistic daily patterns while supporting reproducible training and evaluation.

## Data Source and Preprocessing

A limitation of standard EnergyPlus data-center configurations is that CPU load is typically represented by predefined schedules rather than real workload traces, despite directly affecting server heat generation and electrical demand.

To introduce realistic workload variability, this project uses the [Google Cluster Data 2019](https://github.com/google/cluster-data) traces, which contain CPU-usage data from eight Borg cells collected throughout May 2019 at five-minute intervals.

The data were aggregated by grouping samples into fixed time bins of either one hour or half an hour. In the latter case, the six five-minute samples within each 30-minute window were averaged for every cell and day. The resulting observations were then pooled across eight cells and 31 days to estimate the mean and variance of each interval.
