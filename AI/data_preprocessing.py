"""
Usage Guide
1. Update the '_input_csv_file_path' variable with full path to input csv file,Note: Replace all '\' with '/' in case of windows specific path
2. Update the '_output_csv_file_path' variable with full path to processed output csv file,Note: Replace all '\' with '/' in case of windows specific path
3. Update the 'input_column_names' variable with the column names as per the input dataset,
    Note: Need to define column names as follows
        1. Current
        2. Voltage
        3. Amp-Hours
        4. Temperature
        5. Mode of operation
    The order of list has to be retained 
4. Update the 'output_column_names' variable with the column names to be used for output dataset
    Note: Need to define column names as follows
        1. Current
        2. Voltage
        3. Amp-Hours
        4. Temperature
        5. Mode of operation
    The order of list has to be retained 
5. Update the 'chg_key_word' and 'dis_key_word' with the charge and discharge keywords based on the value from dataset
6. Update the 'cycle_type' keyword with the column name where the charge or discharge is defined
"""

"""
Import Pandas for selecting columns from csv file and read its contents
Import Numpy for manipulating matrixes
"""
import pandas as pd
import numpy as np

"""
_input_csv_file_path contains the complete path to battery dataset csv file
_output_csv_file_path contains the complete path to the processed dataset output
"""
_input_csv_file_path = "/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/A_Test.csv"
_output_csv_file_path = "/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/soc_processed_data.csv"

"""
input_column_names is a list with all the column names of data to be selected for  preprocessing
"""
input_column_names = ["Current, A","Voltage, V","Amp-Hours, AH","Temperature A1, degC","Mode","Real Time"]

"""
output_column_names is a list with all the column names of data to be used for output data csv file, Note the order of column names should be retained as per input column name order
"""
output_column_names = ["Current","Voltage", "Temperature","SoC Capacity","SoC Percentage"]

"""
chg_key_word and dis_key_word is used to define the charge and discharge keyword as per dataset
"""
chg_key_word = "CHRG"
dis_key_word = "DCHG"

"""
cycle_type is used to define the column where the cycle type(discharge/charge) is defined
"""
cycle_type = "Mode"

#Data is read from csv file based on the column names defined in 'usecols' variable using pandas library
cycle = pd.read_csv(_input_csv_file_path,usecols=input_column_names)
cycles = []

#Charge and discharge data is seperated and stored to a different variable cycle_charge and cycle_discharge respectively
cycle_charge = cycle[(cycle[cycle_type] == chg_key_word)]
cycle_discharge = cycle[(cycle[cycle_type] == dis_key_word)]

#Max capacity of battery is found from the dataset by finding the max value under "Amp-Hours AH" column, Note: need to remove neg sign as discharge is defined as negative
max_cap = max(abs(cycle["Amp-Hours, AH"]))

#SOC capacity during discharge goes down so max_cap+(-discharge amp hours) gives the discharge SOC capacity and store it in the same list variable 
cycle_discharge["SoC Capacity"] = max_cap + cycle_discharge["Amp-Hours, AH"]
#SOC in % is calculated using the formula (SOC capacity during discharge in Ah) / (Max capacity during discharge) and store it in the same list variable
cycle_discharge["SoC Percentage"] = cycle_discharge["SoC Capacity"] / max(cycle_discharge["SoC Capacity"])

#SOC capacity during charge increases so charge amp hours gives the charge SOC capacity and store it in the same list variable
cycle_charge["SoC Capacity"] = cycle_charge["Amp-Hours, AH"]
#SOC in % is calculated using the formula (SOC capacity during charge in Ah) / (Max capacity during charge) and store it in the same list variable
cycle_charge["SoC Percentage"] = cycle_charge["SoC Capacity"] / max(cycle_charge["SoC Capacity"])

#Select the required columns from both charge and discharge variable and convert it to numpy format
discharge_numpy = cycle_discharge[["Current, A","Voltage, V", "Temperature A1, degC","SoC Capacity","SoC Percentage"]].to_numpy()
charge_numpy = cycle_charge[["Current, A","Voltage, V", "Temperature A1, degC","SoC Capacity","SoC Percentage"]].to_numpy()

#Stack both the discharge and charge numpy data using vertical stack library of numpy
cycles =np.vstack((discharge_numpy,charge_numpy))
print(cycles)

#Prepare a pandas dataframe with the stacked data and define column names to be used for the exported data as a list under 'columns' variable
data_out = pd.DataFrame(cycles,columns=output_column_names)
#Export the data to specified csv file
data_out.to_csv(_output_csv_file_path)
print('Done')