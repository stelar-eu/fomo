# Problem description

We have a large number of time series for which we want to create a model that predicts future values.
However, the time taken to retrain all models after an update to the data is too long.
Furthermore, the performance of the models is not as good as we would like it to be.
Therefore, we define the problem as a knapsack problem where we want to construct a **portfolio of models** that
**maximizes the overall performance** of the models while the **total time taken** to perform training + inference is
**below a certain time threshold (i.e., a budget)**.
Formally, given a set of time series $\mathcal{T}$, we want to find a partitioning $P$ of $\mathcal{T}$ into $k$
disjoint subsets $P = p_1, p_2, \ldots, p_k$ such that the expected sum of squared errors of the models trained on $P$;
$M = \{m_1, m_2, \ldots, m_k\}$ at a certain time $t$ is minimized while the total time taken to update the predictions
of (a subset of) models in $M$ is below a certain threshold $\tau$.
We assume the time taken to update the predictions of a model is proportional to the number of data points in the time
series, defined as $\gamma_t$, where $t$ is the number of data points in the time series.
For each model, we can decide to update the model or not, which results in a binary decision variable $r_i \in \{0,
1\}$, where $r_i = 1$ if the model $i \in [1,...,k]$ is updated and $r_i = 0$ otherwise.

We can define the problem as an optimization problem as follows:

$$
\begin{align*}
\text{minimize} & \quad \sum_{i=1}^{k} \sum_{y \in p_i} \left (\hat{y}_{m_i,t+1} - y_{t+1} \right)^2 \\
\text{subject to} & \quad \sum_{i=1}^{k} r_i \cdot \gamma_t \leq \tau \\
& \quad r_i \in \{0, 1\} \quad \forall i \in [1,...,k]
\end{align*}
$$
where $y_{t+1}$ is the true value of the time series $y$ at time $t+1$, $\hat{y}_{m_i,t+1}$ is the predicted value of
the time series $y$ at time $t+1$ by model $m_i$, and $\gamma_t$ is the number of data points in the time series at time
$t$.

# Solution description

*Summary:* The time series are not independent and we want to exploit this fact to reduce the number of models that have to be updated and potentially increase their overall performance.
The solution involves two main steps:

1. **Model selection/management:** Deciding and maintaining the partitioning $P$ based on the correlation between the time series and the past performance of the models.
2. **Model prioritization:** Deciding which models to update based on the previous performance of the models and the time taken to update the model them.

### Model selection/management
For model selection and management, we want to cluster the time series in such a way that time series in the same
cluster are similar and can be modeled by the same model without a significant loss in performance.
Furthermore, in order to maintain the clustering over time, we want to use a hierarchical clustering algorithm that
allows us to split/merge clusters as new data comes in and correlations between time series change.
For this, we will use the ODAC algorithm [1], which is a hierarchical clustering algorithm for streaming data that allows us to continuously update the clustering as new data comes in.
The algorithm has shown to be effective in clustering time series data streams and is able to handle large datasets with high dimensionality.
Furthermore, the algorithm is able to handle missing values in the data, which is crucial for our use case as the data is not complete and we want to be able to handle missing values in the time series.

[1] Rodrigues, Pedro Pereira, Joao Gama, and Joao Pedroso. "Hierarchical clustering of time-series data streams." IEEE transactions on knowledge and data engineering 20.5 (2008): 615-627.

### Model prioritization
For model prioritization, we want to prioritize the models that have the highest expected error in the next time step
and that can be updated within the time budget.
Given the assumption that all time series have the same number of data points and therefore the same time taken to
update the model, we can simply order the models based on their average performance since their last update and update the models until the time budget is exhausted.
Performance can be measured using the RMSE or sMAPE, as described in the evaluation plan.
To ensure that the models and the KPIs used to prioritize them can adapt to changes in the data (e.g., new trends, concept drift), we use a sliding window approach to calculate the performance of the models.
Specifically, we use a time window of the last $w$ time steps to calculate the performance of the models and update the models based on this performance.

# Implementation plan

A. **Model selection/management:**

1. Research different similarity measures for time series and select which one is highly correlated with the performance
   of the (aggregate) models.
2. Implement a hierarchical clustering algorithm that can be updated as new data comes in.

B. **Model prioritization:**

1. Implement a function that orders the models based on their average performance since their last update.
2. Implement a function that updates the models until the time budget is exhausted.

# Results

## Model selection

### Researching different similarity measures

We got the following correlations between different distance measaures and the relative RMSE of the pairwise models,
compared to the RMSE of the singular models:

data/monthly.csv

| Distance measure | Correlation          |
|------------------|----------------------|
| canberra         | 0.3788610441982873   |
| cityblock        | 0.36958054047719485  |
| manhattan        | 0.36958054047719485  |
| braycurtis       | 0.32411599880416836  |
| matching         | 0.22703213618608076  |
| hamming          | 0.22703213618608076  |
| euclidean        | 0.17051521080635185  |
| minkowski        | 0.17051521080635185  |
| correlation      | 0.12681822545964297  |
| rogerstanimoto   | 0.12568760749928326  |
| sokalmichener    | 0.12568760749928326  |
| sokalsneath      | 0.09743160081051401  |
| jaccard          | 0.062223311463793556 |
| dice             | 0.033281971870141494 |
| russellrao       | 0.023635585417895105 |
| chebyshev        | -0.0954483256780341  |
| cosine           | -0.09841115405811926 |
| spearman         | -0.1144260217219194  |
| kendall          | -0.11906835822341912 |
| yule             | -0.23645565209770586 |

data/weekly.csv

| Distance measure | Correlation          |
|------------------|----------------------|
| canberra         | 0.38271550047437575  |
| cityblock        | 0.3147675053326674   |
| manhattan        | 0.3147675053326674   |
| rogerstanimoto   | 0.29816808627708347  |
| sokalmichener    | 0.29816808627708347  |
| braycurtis       | 0.25303823744588966  |
| matching         | 0.2487902914271424   |
| hamming          | 0.2487902914271424   |
| sokalsneath      | 0.1638419999497649   |
| jaccard          | 0.15903262991335873  |
| dice             | 0.14268492851175693  |
| correlation      | 0.13680043328721497  |
| russellrao       | 0.10906184757606621  |
| euclidean        | 0.10635396889336703  |
| minkowski        | 0.10635396889336703  |
| cosine           | -0.05286855641718103 |
| chebyshev        | -0.07234075667848508 |
| yule             | -0.16424623432005359 |
| spearman         | -0.17178688130816644 |
| kendall          | -0.1745329575166431  |

data/weekly_syn_r.csv

| Distance measure | Correlation           |
|------------------|-----------------------|
| dice             | 0.18008520285704066   |
| jaccard          | 0.17569480357165      |
| russellrao       | 0.17317946614760404   |
| rogerstanimoto   | 0.1673467557008437    |
| sokalmichener    | 0.1673467557008437    |
| cosine           | 0.1670684490682503    |
| sokalsneath      | 0.16320769115929903   |
| braycurtis       | 0.15164787101581623   |
| canberra         | 0.13164073350872055   |
| correlation      | 0.035265953432408736  |
| yule             | 0.013076411274566721  |
| matching         | -0.008719965662879053 |
| hamming          | -0.008719965662879053 |
| cityblock        | -0.03480048353923071  |
| manhattan        | -0.03480048353923071  |
| euclidean        | -0.03825738134981628  |
| minkowski        | -0.03825738134981628  |
| spearman         | -0.04923959338287775  |
| chebyshev        | -0.05412820472257274  |
| kendall          | -0.05457848783418954  |

Analyzing these results, we can conclude that canberra and manhattan distance are the best options for the real data,
while dice and jaccard are the best options for the synthetic data.
This might tell us that the synthetic data does not have the same properties as the real data.
To be safe, we will go with the results of the real data as that is most likely to be representative of the actual data
of the company.

### Implementing a hierarchical clustering algorithm

For the hierarchical clustering algorithm, it is important that the clustering three is updated as new data comes in.
This means that we need to be able to add new data points to the clustering tree and potentially split or merge clusters
based on the new data.
For this, we used the Online Divisive-Agglomerative Clustering (ODAC)
by [Rodrigues et al. (2008)](https://ieeexplore.ieee.org/document/4407702).
Which is a streaming algorithm which actively checks to see if the clusters need to be split or merged based on the
distances inside a cluster (and its parents).

For implementation, we forked and adapted the code in the paper ([GitHub](https://github.com/rodrigoejcm/odac)) to our
needs and implemented the algorithm in Python.
The result of this is a class named [OdacCluster](/methods/odac_cluster.py) which can be used to cluster time series.

## Model prioritization

For model prioritization, we implemented a series of functions that do the following:

1. Evaluate the predictions on each stream when the new data comes available at a certain time
   $T$ ([fomo.update_forcast_history](/methods/fomo.py)).
2. Group the predictions per model over the last $x$ time steps, to calculate the average error per
   model ([fomo.evaluate_all_models](/methods/fomo.py)).
3. Order the models based on their average error over the last $x$ time
   steps ([fomo.prioritize_updates](/methods/fomo.py)).
4. Update the models until the time budget is exhausted ([fomo.update_forecasts](/methods/fomo.py)).

With this, we can now prioritize the models that need to be updated and update them until the time budget is exhausted.

## Overall implementation

The whole system was implemented in an Object-Oriented way, where the main class is the [FOMO](/methods/fomo.py) class,
and the system is accessed through the main function [main](/main.py).
The system can be run using different strategies for model selection and model prioritization, in order to allow for
comparison of the new system with the old system.
Particularly, we implemented the following model selection strategies:

1. **Singleton**: Each time series is modeled by a separate model. This is the baseline strategy.
2. **ODAC**: The Online Divisive-Agglomerative Clustering algorithm is used to cluster the time series
   and maintain (aggregate) models over them to reduce the number of models that need to be updated.

And the following model prioritization strategies:

1. **Random**: The models are updated in a random order. This is the baseline strategy.
2. **RMSE**: The models are updated in order of their RMSE over the last 10 time steps.
3. **SMAPE**: The models are updated in order of their SMAPE over the last 10 time steps.

To further facilitate the comparison, we measure several metrics during the execution of the system, which are then
saved to a global CSV file in the output directory named `runs.csv` for further analysis.

# Conclusion

The system was implemented and tested on the real and synthetic data.
The results show that the ODAC model selection strategy outperforms the Singleton strategy in terms of RMSE and SMAPE.
Furthermore, the RMSE model prioritization strategy outperforms the Random strategy in terms of RMSE and SMAPE.
The system is now ready to be deployed in the production environment and can be used to update the models in a more
efficient way.

# References

    [1] P. P. Rodrigues, J. Gama and J. Pedroso, "Hierarchical Clustering of Time-Series Data Streams," in IEEE Transactions on Knowledge and Data Engineering, vol. 20, no. 5, pp. 615-627, May 2008, doi: 10.1109/TKDE.2007.190727.




