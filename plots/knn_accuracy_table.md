# CARROT-KNN test-split accuracy

KNN regressor on OpenAI `text-embedding-3-small` embeddings. Metrics on the test split.

| dataset        |   n_test |   n_models |   binarized_acc |   score_MAE |   score_R2 |   routing_acc |   best_single_model_acc |   oracle_acc | cost_MAE   | cost_MAE_unit   |
|:---------------|---------:|-----------:|----------------:|------------:|-----------:|--------------:|------------------------:|-------------:|:-----------|:----------------|
| routerbench    |     7143 |         11 |          0.6884 |      0.3298 |     0.2459 |        0.787  |                  0.7851 |       0.9169 | 0.0001     | $ per query     |
| open-llm-lb-v2 |     4213 |         18 |          0.6701 |      0.4122 |     0.1439 |        0.6143 |                  0.5649 |       0.9715 | N/A        | N/A             |
| sprout         |     6637 |         13 |          0.7943 |      0.2743 |     0.29   |        0.8648 |                  0.8482 |       0.9791 | 357.3613   | output tokens   |
