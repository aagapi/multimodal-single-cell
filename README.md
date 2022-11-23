# Single cell modality prediction competition

This is a description of my solution to the Open Problems - Multimodal Single-cell Integration Kaggle competition (www.kaggle.com/competitions/open-problems-multimodal).
The solution is in the bronze medal area (top 6%, place 70/1220). I feel the approach could have done even better given that:
  1. It uses only the original preprocessed data, rather than raw counts data (I saw that using raw data seemed to provide a major perf boost for the top contenders, so in retrospect it was a mistake to ignore it when it became available)
  2. I focused almost entirely on the Citeseq part, very little on Multiome, where I used a single un-blended model for the submission
  3. Uses very little blending compared to top approaches I saw (only 3 models for Citeseq and 1 for Multiome)
  4. While the feature selection model (SCMER) and the final NNs are pretty well tuned via neural architecture search and hyperparameter tuning, the number of dimensions used in dimensionality reduction (Truncated SVD) was not particularly carefully tuned (I used 64, which is less than other competitors typically used).

Given these aspects, I believe the Citeseq part of this submission could be quite competitive, especially if using raw data.

I think the main 2 contribution areas were in 1) Feature selection; 2) Encoding of batch effects / features, and batch normalization. I will mainly focus on these, and only briefly discuss the other aspects of the solution. Also, I will mainly discuss the Citeseq data, as I have spent limited time on the Multiome dataset.

Overall, my approach was to include as inputs to the model a mix of dimensionality reduced features (embeddings from Truncated SVD from a larger set of features, call it **_S0_**), along with directly including a smaller subset of selected features (call it **_S1_**), un-reduced. In addition, I added features encoding batch effects (discussed at point 2) below). The idea was that the dimred embeddings over **_S0_** would model the example "data cluster" in a lower dimensional space, while smaller selected set of features in **_S1_** would help more directly predict the targets; and the batch features would help predict (rather than remove) batch effects.
A diagram of the overall feature engineering pipeline is shown below.

![pipeline2](https://user-images.githubusercontent.com/16798036/203070988-e0894eb5-aac6-4080-98ef-96711a9abc18.png)


## Feature selection

One difference from most public notebooks and discussions is that I have applied some (quite aggresssive) feature selection even for the features in **_S0_** (not only for **_S1_**), in the hope to further denoise the data before dimensionality reduction. For this, I eliminated features in **_S0_** in 3 stages:

1. Basic Quality Control filtering (min cells threshold where gene is present, min genes in a cell). This reduces the feature set to 21601 features (from the original 22050 in Citeseq)
2. Highly Variable Genes [1][2]. This further reduces the feature set to 1543 features
3. SCMER [3]. Here, I have spent quite some time tuning it, so depending on the choices and parameters described below, this can reduce the S0 feature set to between ~650-1100 features.
 
    3.1. First, we use SCMER in 2 ways, producing a reduced set of features trying to emulate either: 1) the RNA (predictor) manifold - call it **_Y0_** set, or 2) the protein (target) manifold - **_Y1_**. I tried each of these by itself, as well as using the union of both feature sets, i.e. **_Y2 = Union(Y0, Y1)_** as the selected feature set. In general, I found that **_Y2_** outperformed the other 2, and **_Y1_** outperformed **_Y0_**. In the final submission blend, all models used the **_Y2_** set for **_S0_**.

    3.2. Also, I found that SCMER is sensitive to a few parameters, mainly: lasso regularization strength, ridge regularization strength, perplexity, and n_pcs (number of principal components, if using PCA for distances). I tuned these params end-to-end, i.e. picked the ones that produced the best results in cross-validation. The best submitted models used a HVG/SCMER-selected set of size 1051 (of these the **_Y0_** set had 863 features, the **_Y1_** set had 676 features, and there were 488 common features between **_Y0_** and **_Y1_**, thus there were 1051 selected features in the union **_Y2_**). After dimred, this was projected down to 64 dimensions (via Truncated SVD).

For the **_S1_** feature subset (features included directly, with no dimred), I used 2 types of features: 1) features based on name similarity between RNA peaks and proteins; 2) several versions of feature sets picked based on correlation with target proteins; one of the models in the final submission used the set of 84 important features from https://www.kaggle.com/code/pourchot/all-in-one-citeseq-multiome-with-keras, which were also picked by the notebook author based on target correlation). In the final submission blend, I included 1 model using each of these 2 types of **_S1_** set (along with a **_Y2_** **_S0_** set). In general, I found that **_S1_** features based on correlations outperformed features based on name similarity.

Regarding handling of batch effects (detailed below), batch standardization was applied to all features (both **_S0_** and **_S1_**), and for **_S0_** it was applied before Truncated SVD. Batch encoding features were calculated based only on the **_S1_** set.

## Batch normalization and encoding batch effects

Because we want to predict a target modality that is also batch-specific, rather than totally removing batch effects, I tried to also explicitly encode them. Specifically, I first normalize data by batch (removing the batch effects to some degree to facilitate the model cross-learning from multiple batches using the selected features), then explicitly re-encode batch effects using additional batch features. For a choice of batch, I tried **_donor_**, **_day_** and **_'batch' == donor X day_** (i.e. data with same donor and day). In the end, batch-wise models outperformed both donor-wise and day-wise models (but only when batch features are included!). Details are below.

### Batch normalization

After feature selection and before dimensionality reduction, for both S0 and S1 feature sets, I normalize all features using the per-batch means and stdevs of each feature (e.g. mean/stdev of each feature, from all examples with the same donor and day, in the case of batch-wise models). For columns that had a stdev of 0 (sometimes a couple of these were left after feature selection, depending on FS parameters), I only mean-centered the data.

### Batch features

The batch normalization above removes the batch effects to some degree, so I explicitly re-include batch effects in the model by using the batch means for the smaller S1 set of columns, as extra features in the model.
After including batch features, the model performance improved considerably, and I believe batch features and batch normalization were a main contributor to model performance in the end.

Batch features especially helped the performance of models cross-validated using batches (donor X day), versus donor- and day-wise CV models. Specifically, without batch features, batch-wise models tended to overfit in CV and under-perform donor-wise models on the public leaderboard (which had 1 donor not present in the training set, so in principle was best fit for donor-wise validated models). But with batch features included, batch-wise models consistently outperform both donor-wise and day-wise models in all of CV, public LB and private LB.

Besides providing a performance boost (for one, this is because in batch-wise models each fold has much more data to train on, and there are 9 such folds in the data instead of 3), batch features also helped improve correlation between CV and public LB performance, and, as it turned out, enabled the public LB (with an unknown donor) to become a reasonable proxy for the private LB (where all donors were present but there were unknown days, so it was potentially best fit for by-day validated models). In the end, it was a pleasant surprise to see that by-batch validated models using batch features that were beating by-donor and by-day models on the public LB, also did best after the shake-up on the private LB.

## Neural network architecture choices

For the final models, I used neural nets with a custom loss based on Pearson correlation.
After the feature engineering part above, my pipeline included a neural architecture and hyperparameter search module (wiring style between layers, number of layers, layer sizes, batch norm, dropout, regularization terms etc), and produced a best NN for that particular data shape (feature set). The models I used for final submissions had 3 layers and between 128-512 neurons per layer each. Wiring style was resnet-like.

Each model's prediction was an ensemble between all folds of a model (e.g. 9 models in the case of by-batch splitting, 3 models in the case of by-donor). The final submission for Citeseq was an ensemble of 3 models.

Because the competition metric was Pearson correlation, and because predictions from different folds and models potentially had different scales (this mainly happened because models were trained with a Pearson loss, versus something like MSE; and the Pearson loss does not penalize differences in absolute scale between targets and ground truth), before ensembling, outputs from each fold was first standardized using the by-cell means and stdevs. This was done for both ensembling of folds and of different models in the blend. Once the scales became comparable between different folds/models, weighted averaging was applied to produce the final blend.

## Other aspects of the solution

After feature selection and before dimred and building batch features, predictor data (RNA) was rescaled (standardized) using **_scanpy.pp.scale_**. Targets (proteins) were also standardized (this helped with the stability of the Pearson correlation loss, without affecting the comp metric).

## References:

[1] Higly Variable Genes - https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html

[2] Satija et al. (2015), Spatial reconstruction of single-cell gene expression data, Nature Biotechnology, https://www.nature.com/articles/nbt.3192

[3] Liang et al. (2021), Single-cell manifold-preserving feature selection for detecting rare cell populations, Nature Computational Science (2021), https://www.nature.com/articles/s43588-021-00070-7.epdf?sharing_token=niygJFITqVh3dcpF4yT2udRgN0jAjWel9jnR3ZoTv0NZ3tEAbC7cZVb1eDTUaKur6QGQgwC-DQq_MgHYq-ADYRFISGwVSZWO9toKKpTiPePwLal8xfj2okFMXzheGkFzZUumAVNPWrj9qHb4AgVW_B8iRQVjP3RsaFUJN_2YQO8%3D



