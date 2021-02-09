import time
import numpy as np
import random
from sklearn.model_selection import KFold
from numpy import linalg as la
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics.pairwise import cosine_similarity


# Common neighbours count
def Common_Neighbours(adjacency_matrix):
    A = adjacency_matrix
    return np.dot(A, A.T) 


def TS_SVD(drug_disease_matrix, drug_protein_matrix, disease_protein_matrix, latent_feature_percent=0.09):

    none_zero_position = np.where(drug_disease_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]


    ######## code for randomly selected nagative samples
    # zero_position = np.where(drug_disease_matrix == 0)
    # zero_row_index = zero_position[0]
    # zero_col_index = zero_position[1]
    # random.seed(1)
    # zero_random_index = random.sample(range(len(zero_row_index)), len(none_zero_row_index))
    # zero_row_index = zero_row_index[zero_random_index]
    # zero_col_index = zero_col_index[zero_random_index]


    # ######## code for reliable nagative samples in EMP-SVD
    # drug_protein_dis_matrix = np.dot(drug_protein_matrix, disease_protein_matrix.T)
    # zero_deduction_dpd_position = np.where(drug_protein_dis_matrix == 0)
    # zero_deduction_dpd_row_index = zero_deduction_dpd_position[0]
    # zero_deduction_dpd_col_index = zero_deduction_dpd_position[1]
    # random.seed(1)
    # zero_random_index = random.sample(range(len(zero_deduction_dpd_row_index)), len(none_zero_row_index))
    # zero_row_index = zero_deduction_dpd_row_index[zero_random_index]
    # zero_col_index = zero_deduction_dpd_col_index[zero_random_index]


    ######## code for reliable nagative samples in TS-SVD
    # 1-step neighbors: drug->disease
    D1 = drug_disease_matrix
    # 2-step neighbors: drug->protein->disease
    D2 = np.dot(drug_protein_matrix, disease_protein_matrix.T)
    # 3-step neighbors, case 1: drug->protein->drug->disease
    D31 = np.dot(np.dot(drug_protein_matrix, drug_protein_matrix.T), drug_disease_matrix)
    # 3-step neighbors, case 2: drug->disease->drug->disease
    D32 = np.dot(np.dot(drug_disease_matrix, drug_disease_matrix.T), drug_disease_matrix)
    # 3-step neighbors, case 3: drug->disease->protein->disease
    D33 = np.dot(np.dot(drug_disease_matrix, disease_protein_matrix), disease_protein_matrix.T)
    D = D1 + D2 + D31 + D32 + D33
    drug_protein_dis_matrix = D  
    zero_deduction_dpd_position = np.where(drug_protein_dis_matrix == 0)
    zero_deduction_dpd_row_index = zero_deduction_dpd_position[0]
    zero_deduction_dpd_col_index = zero_deduction_dpd_position[1]
    random.seed(1)
    zero_random_index = random.sample(range(len(zero_deduction_dpd_row_index)), len(none_zero_row_index))
    zero_row_index = zero_deduction_dpd_row_index[zero_random_index]
    zero_col_index = zero_deduction_dpd_col_index[zero_random_index]





    row_index = np.append(none_zero_row_index, zero_row_index)
    col_index = np.append(none_zero_col_index, zero_col_index)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    metric_avg = np.zeros((1, 7), float)
    count = 1

    tprs = []
    precisions = []
    auprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 30)
    mean_recall = np.linspace(0, 1, 30)

    for train, test in kf.split(row_index):
        # print('begin cross validation experiment ' + str(count) + '/' + str(kf.n_splits))
        count += 1
        train_drug_disease_matrix = np.copy(drug_disease_matrix)
        test_row = row_index[test]
        test_col = col_index[test]
        train_row = row_index[train]
        train_col = col_index[train]
        np.savetxt('../crossValidation/test_row' + str(count-1) + '.txt', test_row, fmt='%d')
        np.savetxt('../crossValidation/test_col' + str(count-1) + '.txt', test_col, fmt='%d')

        train_drug_disease_matrix[test_row, test_col] = 0
        np.savetxt('../crossValidation/train_drug_disease_matrix_' + str(count-1) + '.txt', train_drug_disease_matrix, fmt='%d')


        #############################################################################################################
        # #### step1: define topological similarity

        # 1. Common neighbours count
        cnDrugSim = Common_Neighbours(train_drug_disease_matrix)
        cnDiseaseSim = Common_Neighbours(train_drug_disease_matrix.T)

        cn_drug_protein_matrix = Common_Neighbours(drug_protein_matrix)
        cn_disease_protein_matrix = Common_Neighbours(disease_protein_matrix)

        cnDrug = cnDrugSim + cn_drug_protein_matrix
        cnDisease = cnDiseaseSim + cn_disease_protein_matrix        

        # 2. Common neighbours count  --> Topolocal Similarity by using cosine similarity measure
        cnDrug = cosine_similarity(cnDrug)
        cnDisease = cosine_similarity(cnDisease) 



        #############################################################################################################
        #### step 2 extract features by SVD
        # latent_feature_percent = 0.09
        (row, col) = train_drug_disease_matrix.shape    
        latent_feature_num1 = int(row * latent_feature_percent)
        latent_feature_num2 = int(col * latent_feature_percent)
        
        (row3, col3) = drug_protein_matrix.shape
        latent_feature_num3 = int(row3 * latent_feature_percent)      
        (row4, col4) = disease_protein_matrix.shape
        latent_feature_num4 = int(row4 * latent_feature_percent)    

        ## using SVD 
        U_Drug, Sigma_Drug, VT_Drug = la.svd(cnDrug)        
        Sigma_Drug = np.diag(Sigma_Drug)

        F_Drug = np.dot(U_Drug, np.sqrt(Sigma_Drug))
        # F_Drug = np.dot( VT_Drug.T, np.sqrt(Sigma_Drug) )                 
        cn_drug_feature_matrix = F_Drug[:, :latent_feature_num1]  


        U_Disease, Sigma_Disease, VT_Disease = la.svd(cnDisease)
        Sigma_Disease = np.diag(Sigma_Disease)

        F_Disease = np.dot( U_Disease, np.sqrt(Sigma_Disease) ) 
        # F_Disease = np.dot( VT_Disease.T, np.sqrt(Sigma_Disease) )             
        cn_disease_feature_matrix = F_Disease[:, :latent_feature_num2]
     

 
        ##########################################################################################################
        #### step 3: construct training dataset and testing dataset

        cn_train_feature_matrix = []   

        train_label_vector = []
        for num in range(len(train_row)):
            feature_vector = np.append(cn_drug_feature_matrix[train_row[num], :],
                                       cn_disease_feature_matrix[train_col[num], :])
            cn_train_feature_matrix.append(feature_vector)
            train_label_vector.append(drug_disease_matrix[train_row[num], train_col[num]])

        cn_test_feature_matrix = []     

        test_label_vector = []
        for num in range(len(test_row)):
            feature_vector = np.append(cn_drug_feature_matrix[test_row[num], :],
                                       cn_disease_feature_matrix[test_col[num], :])
            cn_test_feature_matrix.append(feature_vector)
            test_label_vector.append(drug_disease_matrix[test_row[num], test_col[num]])

        cn_train_feature_matrix = np.array(cn_train_feature_matrix)
        cn_test_feature_matrix = np.array(cn_test_feature_matrix)

        train_label_vector = np.array(train_label_vector)
        test_label_vector = np.array(test_label_vector)


        #################################################################################################
        #### step 4: training and testing
        # here, using Random Forest as an example
        clf1 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)


        # print("training ...")
        clf1.fit(cn_train_feature_matrix, train_label_vector)
        # print("testing ...")
        predict_y_proba = clf1.predict_proba(cn_test_feature_matrix)[:, 1]
        predict_y = clf1.predict(cn_test_feature_matrix)
        # print("evaluating ...")
        AUPR = average_precision_score(test_label_vector, predict_y_proba)
        AUC = roc_auc_score(test_label_vector, predict_y_proba)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(predict_y_proba)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
        metric_avg[0, :] = metric_avg[0, :] + metric


        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        precisions.append(interp(mean_recall, recall, precision))
        auprs.append(AUPR)
        aucs.append(AUC)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sum(aucs) / 5
    
    mean_precision = np.mean(precisions, axis=0)
    mean_aupr = sum(auprs) / 5

    ####### metrics while using randomly selected negative
    # np.savetxt('mean_fpr_TSSVD_random', mean_fpr)
    # np.savetxt('mean_tpr_TSSVD_random', mean_tpr)
    # np.savetxt('mean_precision_TSSVD_random', mean_precision)
    # np.savetxt('mean_recall_TSSVD_random', mean_recall)
    # np.savetxt('mean_auc_aupr_TSSVD_random', (mean_auc, mean_aupr))    

    ####### metrics while using reliable negative samples in EMP-SVD    
    # np.savetxt('mean_fpr_TSSVD_reliableAsEMPSVD', mean_fpr)
    # np.savetxt('mean_tpr_TSSVD_reliableAsEMPSVD', mean_tpr)
    # np.savetxt('mean_precision_TSSVD_reliableAsEMPSVD', mean_precision)
    # np.savetxt('mean_recall_TSSVD_reliableAsEMPSVD', mean_recall)
    # np.savetxt('mean_auc_aupr_TSSVD_reliableAsEMPSVD', (mean_auc, mean_aupr))

    ####### metrics while using reliable negative samples in TS-SVD
    np.savetxt('mean_fpr_TSSVD_reliable', mean_fpr)
    np.savetxt('mean_tpr_TSSVD_reliable', mean_tpr)
    np.savetxt('mean_precision_TSSVD_reliable', mean_precision)
    np.savetxt('mean_recall_TSSVD_reliable', mean_recall)
    np.savetxt('mean_auc_aupr_TSSVD_reliable', (mean_auc, mean_aupr))    

    print("**********************************************************************************************")
    print("AUPR AUC PRE REC ACC MCC F1")
    print(metric_avg / kf.n_splits)
    print("**********************************************************************************************")

if __name__ == "__main__":
    drug_disease_matrix = np.loadtxt('../data/drugDiseaseInteraction.txt', delimiter='\t', dtype=int)
    drug_protein_matrix = np.loadtxt('../data/drugProteinInteraction.txt', delimiter='\t', dtype=int)
    disease_protein_matrix = np.loadtxt('../data/diseaseProteinInteraction.txt', delimiter='\t', dtype=int)
    # drug_similarity_matrix = np.loadtxt('../data/DrugSimilarity.txt', delimiter='\t', dtype=float)
    # protein_similarity_matrix = np.loadtxt('../data/proteinSimilarity.txt', delimiter='\t', dtype=float)
    # disease_similarity_matrix = np.loadtxt('../data/diseaseSimilarity.txt', delimiter='\t', dtype=float)

    # for latent_feature_percent in np.arange(0.01,0.21,0.01):    
    latent_feature_percent = 0.09 
    print()
    print('latent_feature_percent=%s' % (str(latent_feature_percent))  )
    start = time.clock()
    TS_SVD(drug_disease_matrix, drug_protein_matrix, disease_protein_matrix, latent_feature_percent)

    end = time.clock()
    print('Runing time:\t%s\tseconds' % (end - start))
