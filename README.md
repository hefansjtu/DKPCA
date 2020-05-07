# DKPCA
DISTRIBUTED KERNEL PRINCIPAL COMPONENT ANALYSIS in vertical regime, where each local machines contains full samples with only a subset of features.

This is the experiment code of http://arxiv.org/abs/2005.02664, including:

expA:

    expA_NTime_L_sim.m: DKPCA in linear cases with simulation data. Variable is the sample size. Both error and running time is reported.  
    expA_JTime_L_sim.m: DKPCA in linear cases with simulation data. Variable is the number of local machines. Both error and running time is reported.    
    expA_NTime_K_GSE2187.m: DKPCA in non-linear cases with simulation data. Variable is the sample size. Both error and running time is reported.
    expA_JTime_K_GSE2187.m: DKPCA in non-linear cases with simulation data. Variable is the number of local machines. Both error and running time is reported.

expB:

    expB.m: compare DKPCA with DPCA in linear cases with simulation data.

expC:

    expC_GSE2187.m: classification task using DKPCA/KPCA/PCA + linear SVM on GSE2187 dataset.
    
Eigenvectors computed by SVD is regarded as ground truth.
