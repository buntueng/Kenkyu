import os
import cv2
import numpy as np
from fast_glcm import fast_glcm_ASM, fast_glcm_contrast, fast_glcm_dissimilarity, fast_glcm_entropy, fast_glcm_homogeneity, fast_glcm_max, fast_glcm_mean, fast_glcm_std

base_path = os.path.join("D:","Research","CNMC2019")
source_folder = os.path.join(base_path,"preprocessing")


glcm_ams_flag = False
glcm_contras_flag = True
glcm_dissimilarity_flag = False
glcm_energy_flag = False
glcm_entropy_flag = False
glcm_homogeneity_flag = False
glcm_max_flag = False
glcm_mean_flag = False
glcm_standard_flag = False

# create GLCM Standard features
if glcm_standard_flag:
    GLCM_STD_folder = os.path.join(base_path,"GLCM_Standard")
    # if not exist, create the folder
    if not os.path.exists(GLCM_STD_folder):
        os.makedirs(GLCM_STD_folder)
    # convert all images to GLCM Standard
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R = fast_glcm_std(image[:, :, 0])
                glcm_G = fast_glcm_std(image[:, :, 1])
                glcm_B = fast_glcm_std(image[:, :, 2])
                Std_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                Std_glcm = Std_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_STD_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Std_glcm)

# create GLCM Mean features
if glcm_mean_flag:
    GLCM_MEAN_folder = os.path.join(base_path,"GLCM_Mean")
    # if not exist, create the folder
    if not os.path.exists(GLCM_MEAN_folder):
        os.makedirs(GLCM_MEAN_folder)
    # convert all images to GLCM Mean
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R = fast_glcm_mean(image[:, :, 0])
                glcm_G = fast_glcm_mean(image[:, :, 1])
                glcm_B = fast_glcm_mean(image[:, :, 2])
                Mean_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                Mean_glcm = Mean_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_MEAN_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Mean_glcm)

# create GLCM Max features
if glcm_max_flag:
    GLCM_MAX_folder = os.path.join(base_path,"GLCM_Max")
    # if not exist, create the folder
    if not os.path.exists(GLCM_MAX_folder):
        os.makedirs(GLCM_MAX_folder)
    # convert all images to GLCM Max
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R = fast_glcm_max(image[:, :, 0])
                glcm_G = fast_glcm_max(image[:, :, 1])
                glcm_B = fast_glcm_max(image[:, :, 2])
                Max_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                Max_glcm = Max_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_MAX_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Max_glcm)

# create GLCM Homogeneity features
if glcm_homogeneity_flag:
    GLCM_HOMOGENEITY_folder = os.path.join(base_path,"GLCM_Homogeneity")
    # if not exist, create the folder
    if not os.path.exists(GLCM_HOMOGENEITY_folder):
        os.makedirs(GLCM_HOMOGENEITY_folder)
    # convert all images to GLCM Homogeneity
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R = fast_glcm_homogeneity(image[:, :, 0])
                glcm_G  = fast_glcm_homogeneity(image[:, :, 1])
                glcm_B  = fast_glcm_homogeneity(image[:, :, 2])
                Homogeneity_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                Homogeneity_glcm = Homogeneity_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_HOMOGENEITY_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Homogeneity_glcm)

# create GLCM Entropy features
if glcm_entropy_flag:
    GLCM_ENTROPY_folder = os.path.join(base_path,"GLCM_Entropy")
    # if not exist, create the folder
    if not os.path.exists(GLCM_ENTROPY_folder):
        os.makedirs(GLCM_ENTROPY_folder)
    # convert all images to GLCM Entropy
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file)) 
                glcm_R = fast_glcm_entropy(image[:, :, 0])
                glcm_G = fast_glcm_entropy(image[:, :, 1])
                glcm_B = fast_glcm_entropy(image[:, :, 2])
                Entropy_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                Entropy_glcm = Entropy_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_ENTROPY_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Entropy_glcm)

# create GLCM Energy features
if glcm_energy_flag:
    GLCM_ENERGY_folder = os.path.join(base_path,"GLCM_Energy")
    # if not exist, create the folder
    if not os.path.exists(GLCM_ENERGY_folder):
        os.makedirs(GLCM_ENERGY_folder)
    # convert all images to GLCM Energy
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R,glcm_energy_R = fast_glcm_ASM(image[:, :, 0])
                glcm_G,glcm_energy_G  = fast_glcm_ASM(image[:, :, 1])
                glcm_B,glcm_energy_B  = fast_glcm_ASM(image[:, :, 2])
                Energy_glcm = np.dstack((glcm_energy_R, glcm_energy_G, glcm_energy_B)) * 255
                Energy_glcm = Energy_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_ENERGY_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Energy_glcm)
                

# create GLCM Dissimilarity features
if glcm_dissimilarity_flag:
    GLCM_Dissimilarity_folder = os.path.join(base_path,"GLCM_Dissimilarity")
    # if not exist, create the folder
    if not os.path.exists(GLCM_Dissimilarity_folder):
        os.makedirs(GLCM_Dissimilarity_folder)
    # convert all images to GLCM Dissimilarity
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R = fast_glcm_dissimilarity(image[:, :, 0])
                glcm_G = fast_glcm_dissimilarity(image[:, :, 1])
                glcm_B = fast_glcm_dissimilarity(image[:, :, 2])
                Dissimilarity_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                Dissimilarity_glcm = Dissimilarity_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_Dissimilarity_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), Dissimilarity_glcm)

# create GLCM Contras features
if glcm_contras_flag:
    GLCM_CONTRAS_folder = os.path.join(base_path,"GLCM_CONTRAS")
    # if not exist, create the folder
    if not os.path.exists(GLCM_CONTRAS_folder):
        os.makedirs(GLCM_CONTRAS_folder)
    # convert all images to GLCM CONTRAS
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R = fast_glcm_contrast(image[:, :, 0])
                glcm_G  = fast_glcm_contrast(image[:, :, 1])
                glcm_B  = fast_glcm_contrast(image[:, :, 2])
                CONTRAS_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                CONTRAS_glcm = CONTRAS_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_CONTRAS_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), CONTRAS_glcm)
                
# create GLCM AMS features
if glcm_ams_flag:
    GLCM_AMS_folder = os.path.join(base_path,"GLCM_AMS")
    # if not exist, create the folder
    if not os.path.exists(GLCM_AMS_folder):
        os.makedirs(GLCM_AMS_folder)
    # convert all images to GLCM AMS
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                image = cv2.imread(os.path.join(root, file))
                glcm_R,glcm_energy_R = fast_glcm_ASM(image[:, :, 0])
                glcm_G,glcm_energy_G  = fast_glcm_ASM(image[:, :, 1])
                glcm_B,glcm_energy_B  = fast_glcm_ASM(image[:, :, 2])
                ASM_glcm = np.dstack((glcm_R, glcm_G, glcm_B)) * 255
                ASM_glcm = ASM_glcm.astype(np.uint8)
                # check subdirectory is exist or not
                save_dir = os.path.join(GLCM_AMS_folder, root.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file), ASM_glcm)
            