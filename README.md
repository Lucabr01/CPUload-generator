# CPUload Generator
An infinite CPU load trace generator for EnergyPlus environments, designed for integration with Sinergym. It introduces stochastic electrical loads based on statistics extracted from real CPU data, overcoming the deterministic load profiles commonly used in HVAC control simulations. The generated traces preserve realistic daily patterns while supporting reproducible training and evaluation.

## Data Source and Preprocessing

A limitation of standard EnergyPlus data-center configurations is that CPU load is typically represented by predefined schedules rather than real workload traces, despite directly affecting server heat generation and electrical demand.

To introduce realistic workload variability, this project uses the [Google Cluster Data 2019](https://github.com/google/cluster-data) traces, which contain CPU-usage data from eight Borg cells collected throughout May 2019. The observations were organized according to their position within the day and the week, and the corresponding mean and variance were estimated across all cells and available days. This preprocessing preserves both recurring daily patterns and broader weekly trends present in the original workload traces.
