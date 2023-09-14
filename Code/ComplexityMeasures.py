######## Script to obtain all measures


# import os
import pandas as pd
import numpy as np
from Code.Hostility_measure_algorithm import hostility_measure
from Code.measures import ClassificationMeasures
# from sklearn import preprocessing



# root_path = os.getcwd()


# def all_measures(data,save_csv,path_to_save, name_data):
def all_measures(data,name_data):

    # Hostility measure
    y = data['y'].to_numpy()
    X = data.iloc[:, 0:-1].to_numpy()
    sigma = 5
    delta = 0.5
    seed = 0
    k_min = 0
    host_instance, data_clusters, results, k_auto = hostility_measure(sigma, X, y, delta, k_min, seed=0)
    host_instances = np.array(host_instance[k_auto])
    class_data_host = results.loc[k_auto]['Host_0':'Dataset_Host']
    df_class_data_host = pd.DataFrame(class_data_host)
    df_class_data_host.columns = [name_data]





    p = ClassificationMeasures(data)
    kdn = p.k_disagreeing_neighbors()

    DS = p.disjunct_size()
    DCP = p.disjunct_class_percentage()
    TD_U = p.tree_depth_unpruned()
    TD_P = p.tree_depth_pruned()
    MV = p.minority_value()
    CB = p.class_balance()
    CLD = p.class_likeliood_diff()
    N1 = p.borderline_points()  # N1
    N2 = p.intra_extra_ratio()  # N2
    LSC = p.local_set_cardinality()
    LSradius = p.ls_radius()
    H = p.harmfulness()
    U = p.usefulness()
    F1 = p.f1()
    F2 = p.f2()
    F3 = p.f3()
    F4 = p.f4()

    dict_measures = {'Hostility': host_instances, 'kDN': kdn, 'DS': DS, 'DCP': DCP,
                     'TD_U': TD_U, 'TD_P': TD_P, 'MV': MV, 'CB': CB, 'CLD': CLD, 'N1': N1, 'N2': N2,
                     'LSC': LSC, 'LSradius': LSradius, 'H': H, 'U': U, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4,'y':y}

    df_measures = pd.DataFrame(dict_measures)

    # Values per class and dataset
    df_classes_dataset = pd.DataFrame(df_measures.groupby('y').mean())
    df_classes_dataset.loc["dataset"] = df_measures.mean()[:-1]
    df_classes_dataset['Hostility'] = np.array(class_data_host)

    # if (save_csv == True):
    #     # To save the results
    #     os.chdir(path_to_save)
    #     nombre_csv = 'ComplexityMeasures_InstanceLevel_' + name_data + '.csv'
    #     df_measures.to_csv(nombre_csv, encoding='utf_8_sig')
    #
    #     nombre_csv2 = 'ComplexityMeasures_ClassDatasetLevel_' + name_data + '.csv'
    #     df_class_data_host.to_csv(nombre_csv2, encoding='utf_8_sig')

    return df_measures, df_classes_dataset


