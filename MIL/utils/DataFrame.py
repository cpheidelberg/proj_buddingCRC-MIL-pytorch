

#%%
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

class ExcelDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add custom initialization code here if needed

    @classmethod
    def from_excel(cls, excel_file, sheet_name=0):
        # Load an Excel file into the DataFrame
        if isinstance(sheet_name, int):
            # If sheet_name is an integer, assume it's an index
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            except Exception as e:
                raise ValueError(f"Error loading sheet {sheet_name}: {str(e)}")
        elif isinstance(sheet_name, str):
            # If sheet_name is a string, assume it's a sheet name
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            except Exception as e:
                raise ValueError(f"Error loading sheet '{sheet_name}': {str(e)}")
        else:
            raise ValueError("Invalid sheet_name. Please provide an integer or string.")

        # Create an instance of the ExcelDataFrame class with the loaded data
        instance = cls(df)
        return instance

    def calculate_acc(self):
        return accuracy_score(self.label_gt, self.label_pred)

    def calculate_F1(self):
        return f1_score(self.label_gt, self.label_pred)

    def calculate_kappa(self):
        return cohen_kappa_score(self.label_gt, self.label_pred)

    def calculate_tpr_fpr(self):
        '''
        Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

        Args:
            y_real: The list or series with the real classes
            y_pred: The list or series with the predicted classes

        Returns:
            tpr: The True Positive Rate of the classifier
            fpr: The False Positive Rate of the classifier
        '''
        y_real = self.label_gt
        y_pred = self.label_pred
        # Calculates the confusion matrix and recover each element
        cm = confusion_matrix(y_real, y_pred)
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]

        # Calculates tpr and fpr
        tpr = TP / (TP + FN)  # sensitivity - true positive rate
        fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

        return tpr, fpr

    def get_cm(self):
        '''
        calculate the confusion matrix
        Returns: cm
        '''
        y_real = self.label_gt
        y_pred = self.label_pred
        # Calculates the confusion matrix and recover each element
        cm = confusion_matrix(y_real, y_pred)
        return cm

    def get_all_roc_coordinates(self):
        '''
        Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.

        Args:
            y_real: The list or series with the real classes.
            y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

        Returns:
            tpr_list: The list of TPRs representing each threshold.
            fpr_list: The list of FPRs representing each threshold.
        '''
        y_proba = self.label_prob
        y_real = self.label_gt
        tpr_list = [0]
        fpr_list = [0]
        for i in range(len(y_proba)):
            threshold = y_proba[i]
            y_pred = y_proba >= threshold
            tpr, fpr = self.calculate_tpr_fpr(y_real, y_pred)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        return tpr_list, fpr_list

    def plot_roc_curve(tpr, fpr, name, auc, scatter=True, ax=None):
        '''
        Plots the ROC Curve by using the list of coordinates (tpr and fpr).

        Args:
            tpr: The list of TPRs representing each coordinate.
            fpr: The list of FPRs representing each coordinate.
            name: (str) name of the classification to be displayed in the label
            auc:  (float) AUC value to be displayed alongside the classification inside the label
            scatter: When True, the points used on the calculation will be plotted with the line (default = True).
        '''
        if ax == None:
            plt.figure(figsize=(5, 5))
            ax = plt.axes()

        if scatter:
            sns.scatterplot(x=fpr, y=tpr, ax=ax)
        sns.lineplot(x=fpr, y=tpr, ax=ax, legend='brief', label=f'{name} (AUC={auc:0.3f})', alpha=0.8)

        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
        # ax.legend()

    def calculate_AUC(self):
        roc_auc_ovr = roc_auc_score(self.label_gt, self.label_prob)
        return roc_auc_ovr

#%%
#df = ExcelDataFrame.from_excel("../plots/df_TAG#xrmATxjNy0.xlsx")

