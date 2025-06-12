from __future__ import annotations
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any




"""

Very important note to avoid buggy behavior in case of modification.
the set predictions are coded so that:
- For the case of MCC, a set prediction is a numpy array, e.g., array([0, 7]

- For the case of MLC or MDC, a set prediction is a **list** of numpy arrays, e.g.,

[array([0, 1]),
 array([0]),
 array([0, 1]),
 array([0]),
 array([0, 1]),
 array([0, 1])]

 or 

 [array([4]),
 array([1, 2]),
 array([3]),
 array([4]),
 array([3]),
 array([2]),
 array([4]),
 array([3])]
 


"""



class Generic(ABC):

    def __init__(self, dataset):
        # self.__class__.__name__ gets the name of the subclass (e.g., 'MCC', 'MDC')
        class_name = self.__class__.__name__
        file_path = f'cache/{dataset}_{class_name}.npz'

        # Common data loading and attribute assignment
        all_data = np.load(file_path, allow_pickle=True)
        self.probas = all_data['probas']
        self.y = all_data['y']
        self.fold_indices = all_data['fold_indices'].tolist()
        self.classes = all_data['classes']
      
    metrics = [
        'u_alpha',
        'recall',
        'correct singleton prediction',
        'correct set-valued prediction',
        'set-size',
        'singleton prediction'
    ]

    @staticmethod
    def bsc_abstention(probabilistic_predictions, n_instances, outcomes, alpha):
        n_outcomes = len(outcomes) 
        bsc_abtension_predictions = []
        abstention_reward = (alpha + (1-alpha)/(n_outcomes))*(1/(n_outcomes))
        for n in range(n_instances):
            current_probabilistic_prediction = probabilistic_predictions[n]
            class_order = np.argsort(-current_probabilistic_prediction)
            if current_probabilistic_prediction[class_order[0]] >= abstention_reward:
                optimal_cardinality = 1
            else: 
                optimal_cardinality = n_outcomes
            bsc_abtension_predictions.append(outcomes[class_order[0:optimal_cardinality]])
        return bsc_abtension_predictions
    @staticmethod
    def bsc_cautious(probabilistic_predictions, n_instances, outcomes, alpha):
        n_outcomes = len(outcomes) 
        bsc_cautious_predictions = []
        for n in range(n_instances):
            current_probabilistic_prediction = probabilistic_predictions[n]
            total_probability = 0
            max_expected_score = 0
            optimal_cardinality = 0
            class_order = np.argsort(-current_probabilistic_prediction)
            for cardinality in range(n_outcomes):
                g = (alpha + (1-alpha)/(cardinality + 1))*(1/(cardinality + 1))
                total_probability += current_probabilistic_prediction[class_order[cardinality]] 
                expected_score = g * total_probability
                if max_expected_score > expected_score:
                    break
                else: 
                    max_expected_score = expected_score
                    optimal_cardinality += 1
            bsc_cautious_predictions.append(outcomes[class_order[0:optimal_cardinality]])
        return bsc_cautious_predictions
    

    @staticmethod
    def U(alpha, z):
        return (alpha + (1-alpha)/z)*(1/z)
    

    @staticmethod
    def is_in(candidate, sets):
        return all(x in s for x, s in zip(candidate, sets))
    

    @staticmethod   
    def size(x):
        size = 1
        for e in x:
            size *= len(e)
        return size
               

    #this function computes several average scores for an alpha, according to prediction type (cautious or abstention)
    # It is designed for MCC and (local) MDC, where the 'prediction_each_fold' will be overridden accordingly for these cases
    def average_scores(self, prediction_type,alpha):
        y = self.y
        fold_indices = self.fold_indices

        u_alpha_scores = []
        recall_scores = []

        correct_singleton_prediction_scores = []
        correct_set_valued_prediction_scores = []


        set_size_predictions = []
        singleton_prediction_scores = []
        
       
        #for each fold i
        for i in range(len(self.probas)):
            u_alpha = 0
            recall = 0
            set_size = 0
            correct_singleton_prediction = 0
            
            singleton_prediction = 0
            correct_set_valued_prediction = 0
            
            set_valued_prediction = 0
            
            
            
            #this has to be overridden for MCC and (local) MDC
            bsc_predictions = self.prediction_each_fold(prediction_type, self.probas[i],  alpha)
            number_intances = len(bsc_predictions)
            # for each instance in the fold
            for instance_index, instance in enumerate(bsc_predictions):


                truth = y[fold_indices[i]][instance_index]
                #Check if it is the case of MDC or local MLC (because a list of arrays) otherwise it is the case of MCC

                """
                zipped is created to make the code holds for every case
                -In MLC or MDC, instance = [Z1,...., Zk] where each Zk is an array - the associated prediction for each class; and truth is an array 
                Then zipped = (truth_1, Z1),..., (truth_k, Zk)

                -For MCC, truth is a scalar and instance an array 
                Then zipped = (truth, array)
                
                """
                zipped = zip(truth, instance) if isinstance(instance, list) else zip([truth], [instance])
                #number of class variables
                K = len(instance) if isinstance(instance, list) else 1

                for yk, Zk in zipped:
                    is_inside = yk in Zk
                    z_len = len(Zk)
                    if is_inside:
                        u_alpha += Generic.U(alpha, z_len)
                        recall += 1
                    if z_len == 1:
                        singleton_prediction +=1
                        if yk in Zk:
                            correct_singleton_prediction +=1
                    if z_len > 1:
                        set_valued_prediction +=1
                        set_size += z_len
                        if is_inside:
                            correct_set_valued_prediction += 1


            u_alpha_scores.append((u_alpha / number_intances)*1/K)
            recall_scores.append((recall / number_intances)*1/K)
            singleton_prediction_scores.append((singleton_prediction / number_intances)*1/K)

            #No need to put the term 1/K because it is cancelled out from both numerator and denominator
            correct_singleton_prediction_scores.append((correct_singleton_prediction / singleton_prediction) if singleton_prediction > 0 else 1)
            correct_set_valued_prediction_scores.append((correct_set_valued_prediction / set_valued_prediction) if set_valued_prediction > 0 else 1)
            set_size_predictions.append((set_size / set_valued_prediction) if set_valued_prediction > 0 else 0)      

        return {'u_alpha':np.mean(u_alpha_scores), 
            'set-size': np.mean(set_size_predictions),
            'recall':np.mean(recall_scores),
            'correct singleton prediction': np.mean(correct_singleton_prediction_scores), 
            'singleton prediction': np.mean(singleton_prediction_scores),
            'correct set-valued prediction': np.mean(correct_set_valued_prediction_scores) 
            }
    

    def average_scores_abstention(self,alpha):
        return self.average_scores(Generic.bsc_abstention, alpha)

        
    def average_scores_cautious(self,alpha):
        return self.average_scores(Generic.bsc_cautious, alpha)
        
    @abstractmethod
    def prediction_each_fold(self, prediction_type, probas,  alpha):
        """Concrete subclasses *must* override"""

    

    def compute_results_for_dataset(self, alpha_vals: List[float]) -> Dict[str, Dict[str, List[float]]]:
        """
        Runs through all alphas, instantiates self (subclass instance),
        computes average_scores_abstention and average_scores_cautious,
        and returns a nested dict:
            results['abstention'][metric] -> [values over alpha_vals]
            results['cautious'][metric]   -> [values over alpha_vals]
        """
        metrics = self.metrics

        results = {
            'cautious': {m: [] for m in metrics},
            'abstention': {m: [] for m in metrics}
        }

        for alpha in alpha_vals:
            # Update the instance's alpha value
            # Each of these returns a dict mapping metric -> float
            abst_scores = self.average_scores_abstention(alpha)
            caut_scores = self.average_scores_cautious(alpha)

            for m in metrics:
                results['abstention'][m].append(abst_scores[m])
                results['cautious'][m].append(caut_scores[m])

        return results




class MCC(Generic):

    #prediction for each fold will be done by just applying "Generic.bsc_abstention" or "Generic.bsc_cautious"
    def prediction_each_fold(self, prediction_func, probas, alpha):
        return prediction_func(probas, len(probas), self.classes, alpha)
       


class MDC(Generic):

    # For the case of (local) MDC, "Generic.bsc_abstention" ("Generic.bsc_cautious") is applied independently for each class variable then combine the result 
    def prediction_each_fold(self, prediction_func, probas, alpha):
        d = []
        for current_class in range(len(self.classes)):
            d.append(prediction_func(probas[current_class], len(probas[current_class]), self.classes[current_class], alpha))
        #each prediction is the Cartesian product of the predictions for each class variable
        #zip(*d) makes sure that each item in the list corresponds to a single instance
        return [list(items) for items in zip(*d)]


class MLC(MDC):

    def __init__(self, dataset, mode):
        # Call the parent's __init__ to handle the common data loading
        super().__init__(dataset)
        # Handle the specific attribute for MLC: either "local" or "global"
        self.mode = mode


    #this function is for the case of global MLC only. It implements the search algorithm in the paper
    def global_MLC_predictions(self,probas, alpha):
        # predicted probabilities for that each  instance across every possible class
        probas = np.stack(probas, axis=1)
        bsc_cautious_predictions = []
        for P in probas:
            K= P.shape[0]

            #row indices according to the max value of each row when sorted in decreasing order 
            sorted_row_indices = np.argsort(-np.max(P, axis=1))
        
            #the max value of each row when sorted in decreasing order 
            max_values= sorted(np.max(P, axis=1), reverse=True)


            #initially, the optimal set is the one with size 1
            expected_best_score= np.prod(max_values)
            product_probability = np.prod(max_values)
            best_prediction = list(np.argmax(P, axis=1))
            
            for k in range(1,K+1):
                product_probability = product_probability / max_values[-k]
                current_score = product_probability*MLC.U(alpha,2**k)
                if expected_best_score > current_score:
                    break
                
                else:
                    expected_best_score = current_score
                    best_prediction[sorted_row_indices[-k]] = np.array([0,1])

            #turn to list of arrays for constistency with other methods
            best_prediction = [np.array([e]) if np.isscalar(e) else e for e in best_prediction]
            bsc_cautious_predictions.append(best_prediction)
        return bsc_cautious_predictions
   



    #this function is for the case of local MLC only
    def prediction_each_fold(self, prediction_func,probas,alpha):
        
        if self.mode != 'local':
            raise AttributeError("Method 'prediction_each_fold' is only available in 'local' mode.")
            # prediction_func does not matter here, call MDC’s implementation directly because cautious is the same as abstention
        return MDC.prediction_each_fold(self, Generic.bsc_cautious, probas, alpha)
        
    

    #This function is for the case of global MLC only. In contrast to the local MLC, its mechanism to compute the metrics is based on 0-1 utility so it has an own implementation.
    #Can be  overridden for the case of global MDC as well if needed in the future
    def global_average_scores(self,alpha):

        if self.mode != 'global':
            raise AttributeError("Method 'global_average_scores' is only available in 'global' mode.")
        y = self.y
        fold_indices = self.fold_indices

        u_alpha_scores = []
        recall_scores = []

        correct_singleton_prediction_scores = []
        correct_set_valued_prediction_scores = []


        set_size_predictions = []
        singleton_prediction_scores = []
        
       
        #for each fold i
        for i in range(len(self.probas)):
            u_alpha = 0
            recall = 0
            set_size = 0
            correct_singleton_prediction = 0
            
            singleton_prediction = 0
            correct_set_valued_prediction = 0
            
            set_valued_prediction = 0
            
            
            bsc_predictions = self.global_MLC_predictions(self.probas[i], alpha)
            number_intances = len(bsc_predictions)
            # for each instance in the fold
            for instance_index, instance in enumerate(bsc_predictions):
                size_of_instance = MLC.size(instance)

                is_in = MLC.is_in(y[fold_indices[i]][instance_index], instance)
                
                if is_in:
                    u_alpha += MLC.U(alpha, size_of_instance)
                    recall += 1

                if size_of_instance == 1:
                    if is_in:
                        correct_singleton_prediction +=1
                    singleton_prediction +=1

                if size_of_instance > 1:
                    if is_in:
                        correct_set_valued_prediction += 1
                    set_valued_prediction +=1  
                    set_size +=  size_of_instance  

            u_alpha_scores.append(u_alpha / number_intances)
            recall_scores.append(recall / number_intances)
            singleton_prediction_scores.append(singleton_prediction / number_intances)
            correct_singleton_prediction_scores.append(correct_singleton_prediction / singleton_prediction if singleton_prediction > 0 else 1)
            correct_set_valued_prediction_scores.append(correct_set_valued_prediction / set_valued_prediction if set_valued_prediction > 0 else 1)
            set_size_predictions.append(set_size / set_valued_prediction if set_valued_prediction > 0 else 0)      

        return {'u_alpha':np.mean(u_alpha_scores), 
            'set-size': np.mean(set_size_predictions),
            'recall':np.mean(recall_scores),
            'correct singleton prediction': np.mean(correct_singleton_prediction_scores), 
            'singleton prediction': np.mean(singleton_prediction_scores),
            'correct set-valued prediction': np.mean(correct_set_valued_prediction_scores) 
            }
    

    #this function is overridden because cautious = abstention in this case
    
    def compute_results_for_dataset(self, alpha_vals):
        metrics = self.metrics
        results = {'cautious': {m: [] for m in metrics}}

        for alpha in alpha_vals:
            scores = (self.average_scores_cautious(alpha) 
                      if self.mode == "local" else self.global_average_scores(alpha))
            for m in metrics:
                results['cautious'][m].append(scores[m])
        return results

class PlotterGeneric:
    """
    Generic plotter base class.

    Parameters
    ----------
    data_name : str
        Name of the dataset (used in filenames and titles).
    alpha_vals : array-like, shape (N,)
        List or array of alpha values.
    results : dict
        Nested dictionary of results. 
        Expected format: results[method][metric] → list of length N.
    methods : sequence of str
        Names of the methods to plot (e.g. ('cautious', 'abstention')).
    colors : dict, optional
        Mapping from method name to a color string. If not provided, defaults will be used.
    save_dir : str, optional
        Directory in which to save PNG/PDF outputs. Defaults to 'selected_results'.
    """
    def __init__(self,
                 data_name: str,
                 alpha_vals,
                 results: dict,
                 methods=('cautious', 'abstention'),
                 acronyms = ('clf_ca', 'clf_ab'),
                 colors=None,
                 save_dir='selected_results'):
        self.data_name = data_name
        self.alpha_vals = np.array(alpha_vals)
        self.results = results
        self.methods = list(methods)
        self.acronyms = list(acronyms)

        # Default color mapping if none provided
        if colors is None:
            default_colors = {
                'clf_ca': 'r',
                'clf_ab': 'b'
            }
            # Filter default_colors for the methods we actually have
            self.colors = {m: default_colors.get(m, None) for m in self.acronyms}
        else:
            # Use provided colors, but ensure there is a color for each method
            self.colors = {m: colors.get(m, None) for m in self.acronyms}

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_and_close(self, fig, filename_base: str):
        """
        Save the figure in both PDF and PNG formats, then close it.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save.
        filename_base : str
            Base name (without extension) for the saved files.
        """
        pdf_path = os.path.join(self.save_dir, f'{self.data_name}_{filename_base}.pdf')
        png_path = os.path.join(self.save_dir, f'{self.data_name}_{filename_base}.png')
        fig.savefig(pdf_path, format='pdf')
        fig.savefig(png_path, format='png')
        plt.close(fig)

    def plot_set_size(self):
        """
        Plot 'set-size' vs. alpha. Skips the first alpha index for both methods.
        """
        fig = plt.figure(figsize=(5, 4))
        for index in range(len(self.methods)):
            yvals = self.results[self.methods[index]]['set-size']
            plt.plot(
                self.alpha_vals[1:],
                yvals[1:],
                label=f'{self.acronyms[index].capitalize()}:siz',
                linewidth=1,
                color=self.colors.get(self.acronyms[index])
            )

        plt.xlabel('Alpha')
        plt.ylabel('Set-Size'.capitalize())
        plt.title(f'Data set: {self.data_name.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_and_close(fig, 'set-size')

    def plot_singleton_prediction(self):
        """
        Plot 'singleton prediction' vs. alpha over the full range.
        """
        fig = plt.figure(figsize=(5, 4))
        for index in range(len(self.methods)):
            plt.plot(
                self.alpha_vals,
                self.results[self.methods[index]]['singleton prediction'],
                label=f'{self.acronyms[index].capitalize()}:pro',
                linewidth=1,
                color=self.colors.get(self.acronyms[index])
            )

        plt.xlabel('Alpha')
        plt.ylabel('Singleton Prediction')
        plt.title(f'Data set: {self.data_name.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_and_close(fig, 'singleton_prediction')

    def plot_u_alpha_recall(self):
        """
        Plot 'u_alpha' (dashed) and 'recall' (solid) vs. alpha for each method.
        """
        fig = plt.figure(figsize=(5, 4))
        for index in range(len(self.methods)):
            # u_alpha (dashed)
            plt.plot(
                self.alpha_vals,
                self.results[self.methods[index]]['u_alpha'],
                label=f'{self.acronyms[index].capitalize()}:ual',
                linestyle='--',
                linewidth=1,
                color=self.colors.get(self.acronyms[index])
            )
            # recall (solid)
            plt.plot(
                self.alpha_vals,
                self.results[self.methods[index]]['recall'],
                label=f'{self.acronyms[index].capitalize()}:rec',
                linewidth=1,
                color=self.colors.get(self.acronyms[index])
            )

        plt.xlabel('Alpha')
        plt.ylabel('u_alpha (ual) and Recall (rec)')
        plt.title(f'Data set: {self.data_name.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_and_close(fig, 'u_alpha_recall')

    def plot_correct_predictions(self):
        """
        Plot 'correct singleton prediction' (dashed) and 'correct set-valued prediction' (solid),
        skipping the first alpha index for both.
        """
        fig = plt.figure(figsize=(5, 4))
        for index in range(len(self.methods)):
            # correct singleton (dashed)
            plt.plot(
                self.alpha_vals[1:],
                self.results[self.methods[index]]['correct singleton prediction'][1:],
                label=f'{self.acronyms[index].capitalize()}:sin',
                linestyle='--',
                linewidth=1,
                color=self.colors.get(self.acronyms[index])
            )
            # correct set-valued (solid)
            plt.plot(
                self.alpha_vals[1:],
                self.results[self.methods[index]]['correct set-valued prediction'][1:],
                label=f'{self.acronyms[index].capitalize()}:set',
                linewidth=1,
                color=self.colors.get(self.acronyms[index])
            )

        plt.xlabel('Alpha')
        plt.ylabel('Correct Singleton and Set-Valued Prediction')
        plt.title(f'Data set: {self.data_name.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_and_close(fig, 'correct_predictions')

    def plot_all(self):
        """
        Generate all available plots in sequence.
        """
        self.plot_set_size()
        self.plot_singleton_prediction()
        self.plot_u_alpha_recall()
        self.plot_correct_predictions()





class PlotterMLC(PlotterGeneric):
    """
    Multi-label classification plotter (MLC). Inherits from PlotterGeneric,
    but only uses the 'cautious' because in this case both methods are the same.
    """
    def __init__(self,
                 data_name: str,
                 alpha_vals,
                 results: dict,
                 save_dir='selected_results'):
        super().__init__(
            data_name=data_name,
            alpha_vals=alpha_vals,
            results=results,
            methods=('cautious',),
            acronyms = ('clf_ca',),
            # only one method, so just assign a color to 'cautious'
            colors={'clf_ca': 'r'},
            save_dir=save_dir
        )

if __name__ == '__main__':
    # Common alpha values
    alphas = np.linspace(1.0, 3.0, 20)

   
    #The following is for MCC or (local) MDC. Uncomment to use it

    
    dataset_names = ['Pain']

    for name in dataset_names:
         # Compute the results dict for this dataset
         runner = MDC(name)
         res = runner.compute_results_for_dataset(alphas)

         # Create a plotter instance

         plotter = PlotterGeneric(
             data_name=name,
             alpha_vals=alphas,
             results=res,
             methods=('cautious', 'abstention'),
             acronyms = ('clf_ca', 'clf_ab'),
             save_dir='selected_results/MDC'
         )
        
         # plotter = PlotterMLC(data_name=name,alpha_vals=alphas, results=res)


         # Generate and save all four plots
         plotter.plot_all()

    dataset_names = ['scene']

    for name in dataset_names:
        # Compute the results dict for this dataset
        runner = MLC(name,'local')
        res = runner.compute_results_for_dataset(alphas)

        # Create a plotter instance

        plotter = PlotterMLC(
            data_name=name,
            alpha_vals=alphas,
            results=res,
            save_dir=f'selected_results/MLC/{runner.mode}'
        )
        
        # plotter = PlotterMLC(data_name=name,alpha_vals=alphas, results=res)


        # Generate and save all four plots
        plotter.plot_all()
