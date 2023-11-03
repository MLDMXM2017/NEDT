import numpy as np
import pandas as pd
import time


def load_data(in_dir, feature_types=['mf', 'pp']):
    label_file = "%s/DDI/data/DDI label.csv" % in_dir
    finger_file = "%s/DDI/data/fingerprint.csv" % in_dir
    phychem_file = "%s/DDI/data/no_normalized_physicochemical.csv" % in_dir
    # phychem_file = "%s/DDI/data/normalized_physicochemical.csv" % in_dir

    s = time.time()
    AB_L = pd.read_csv(label_file)
    finger = pd.read_csv(finger_file)  # 881
    phychem = pd.read_csv(phychem_file)  # 200

    feature_info = {}
    feature_info['mf'] = (finger.shape[1] - 1) * 2
    feature_info['pp'] = (phychem.shape[1] - 1) * 2

    n_samples = AB_L.shape[0]
    # print("Total samples: {}".format(n_samples))
    n_feature = 1
    for feature in feature_types:
        n_feature += feature_info[feature]
    data = np.zeros((n_samples, n_feature))
    ee = []
    k = 0
    for i in range(n_samples):
        A_id = AB_L.at[i, "Drug1_SMILES"]
        B_id = AB_L.at[i, "Drug2_SMILES"]

        A_finger, B_finger = None, None
        if 'mf' in feature_types:
            A_finger = np.array(finger[finger["Drug_SMILES"] == A_id]).reshape(-1)[1:]
            B_finger = np.array(finger[finger["Drug_SMILES"] == B_id]).reshape(-1)[1:]  # SMILES

            if A_finger.shape[0] == 0:
                if A_id not in ee:
                    ee.append(A_id)
                continue
            if B_finger.shape[0] == 0:
                if A_id not in ee:
                    ee.append(B_id)
                continue

        A_phychem, B_phychem = None, None
        if 'pp' in feature_types:
            A_phychem = np.array(phychem[phychem["SMILES"] == A_id]).reshape(-1)[1:]
            B_phychem = np.array(phychem[phychem["SMILES"] == B_id]).reshape(-1)[1:]

            if A_phychem.shape[0] == 0:
                if A_id not in ee:
                    ee.append(A_id)
                continue
            if B_phychem.shape[0] == 0:
                if A_id not in ee:
                    ee.append(B_id)
                continue

        label = np.array([AB_L.at[i, "label"]])

        items = [A_finger, A_phychem, B_finger, B_phychem, label]
        selected_items = [item for item in items if item is not None]
        sample = np.concatenate(selected_items)

        data[k] = sample
        k += 1

    # print("Load {} valid samples".format(k))
    data = data[0:k, :]
    e = time.time()
    # print("time:{:.2f}min".format((e-s)/60))
    # return data[:,0:-1], data[:,-1]
    return data

# x,y=load_data()
# tmp=np.sum(x,axis=1)
# print(tmp.shape)
# index=np.where(tmp==0)[0]
# print(len(index))



