1. Run the three algorithms in the utils folder
utils/BIQI_release/compute_statistics.m
utils/BRISQUE_release/compute_brisquefeature.m
utils/GM-LOG-BIQA/compute_feature.m

2. Based on the above algorithms, extract image quality features x_q from the Live database, and then train a model to predict the quality score y_q.

3. Using the trained model, predict the quality score y_q.