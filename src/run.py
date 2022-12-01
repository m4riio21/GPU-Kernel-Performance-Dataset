import time
print("[*] Initializing scripts...")
time.sleep(3)
print("[*] Running exploratory_data_analysis module")
import exploratory_data_analysis
print("[*] Running data_preprocessing module")
import data_preprocessing
print("[*] Running model_selection module")
import model_selection
print("[*] Running model_selection_4best module")
import model_selection_4best
print("[*] Running optimal_number_of_features module")
import optimal_number_of_features
print("[*] Running model_selection_optimalattributes module")
import model_selection_optimalattributes
print("[*] Running hyperparameter_search module")
import hyperparameter_search
print("[*] All modules have been run, you can check the results in results/ dir")
time.sleep(10)
