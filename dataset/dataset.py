import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from torch.utils.data import Dataset

USE_INCLUDED_FILE = "USE_INCLUDED_FILE"
thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")


def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample


class CheX_Dataset(Dataset):
    """CheXpert Dataset

    Citation:

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison. Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko,
    Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
    Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
    Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng.
    https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well, but is so tiny,
    it is not included here.
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        views=["PA"],
        transform=None,
        data_aug=None,
        flat_dir=True,
        seed=0,
        unique_patients=True,
    ):
        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "chexpert_train.csv.gz")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv[
            "AP/PA"
        ]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace(
            {"Lateral": "L"}
        )  # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r"(patient\d+)")
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # add consistent csv values

        # offset_day_int

        # patientid
        if "train" in self.csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif "valid" in self.csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid

        # age
        self.csv["age_years"] = self.csv["Age"] * 1.0
        self.csv["Age"][(self.csv["Age"] == 0)] = None

        # sex
        self.csv["sex_male"] = self.csv["Sex"] == "Male"
        self.csv["sex_female"] = self.csv["Sex"] == "Female"

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(
            len(self), self.views, self.data_aug
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["Path"].iloc[idx]
        # clean up path in csv so the user can specify the path
        imgid = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = img
        #sample = apply_transforms(sample, self.transform)
        #sample = apply_transforms(sample, self.data_aug)

        return sample
    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view