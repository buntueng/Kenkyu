import os
import pandas as pd
subfolder_list = ['0_db_fan','6_db_fan','m6_db_fan']
dataset_path = 'D:/Research/Audio_FaultDetection/MIMII/dataset'
current_folder = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_folder, 'fan_dataset.csv')

# read all files in the dataset and create a list of paths and label
def create_path_and_label(dataset_path,subfolder_list):
    path_list = []
    label_list = []
    for subfolder in subfolder_list:
        dataset_path = os.path.join(dataset_path, subfolder)
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    path_list.append(os.path.join(root, file))
                    label = os.path.basename(root).split('/')[-1]
                    if label == 'normal':
                        label_list.append(0)
                    else:
                        label_list.append(1)
    return path_list, label_list


path_list, label_list = create_path_and_label(dataset_path,subfolder_list)
# save path and label list to a csv file

df = pd.DataFrame(list(zip(path_list, label_list)), columns=['path', 'label'])
df.to_csv(output_file, index=False)