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
import matplotlib.pyplot as plt

"""
_input_csv_file_path contains the complete path to battery dataset csv file
_output_csv_file_path contains the complete path to the processed dataset output
"""
_input_csv_file_path = "/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/A_Test.csv"
_output_csv_file_path = "/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/soc_processed_data.csv"

"""
Rated Capacity of Battery as per manufacturer in AH
"""
battery_rated_capacity = 20
"""
input_column_names is a list with all the column names of data to be selected for  preprocessing
"""
input_column_names = ["Step","Current, A","Voltage, V","Amp-Hours, AH","Temperature A1, degC","Mode","Real Time"]

"""
output_column_names is a list with all the column names of data to be used for output data csv file, Note the order of column names should be retained as per input column name order
"""
output_column_names = ["Current","Voltage", "Temperature","SoC Capacity","SoC Percentage","SoH Percentage"]

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
cycle = pd.read_csv(_input_csv_file_path,usecols=input_column_names,na_values="NaN")
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
#SOH Discharge is calculated using the formula (Battery capacity)/Rated capacity of battery
cycle_discharge["SoH Percentage"] = cycle_discharge["SoC Capacity"] / battery_rated_capacity
#SOC capacity during charge increases so charge amp hours gives the charge SOC capacity and store it in the same list variable
cycle_charge["SoC Capacity"] = cycle_charge["Amp-Hours, AH"]
#SOC in % is calculated using the formula (SOC capacity during charge in Ah) / (Max capacity during charge) and store it in the same list variable
cycle_charge["SoC Percentage"] = cycle_charge["SoC Capacity"] / max(cycle_charge["SoC Capacity"])
#SOH Discharge is calculated using the formula (Battery capacity)/Rated capacity of battery
cycle_charge["SoH Percentage"] = cycle_charge["SoC Capacity"] / battery_rated_capacity

#Select the required columns from both charge and discharge variable and convert it to numpy format
discharge_numpy = cycle_discharge[["Current, A","Voltage, V", "Temperature A1, degC","SoC Capacity","SoC Percentage","SoH Percentage"]].to_numpy()
charge_numpy = cycle_charge[["Current, A","Voltage, V", "Temperature A1, degC","SoC Capacity","SoC Percentage","SoH Percentage"]].to_numpy()

#Stack both the discharge and charge numpy data using vertical stack library of numpy
cycles =np.vstack((discharge_numpy,charge_numpy))
print(cycles)

#Prepare a pandas dataframe with the stacked data and define column names to be used for the exported data as a list under 'columns' variable
data_out = pd.DataFrame(cycles,columns=output_column_names)
#Export the data to specified csv file
data_out.to_csv(_output_csv_file_path)
print('Done')

"""
Plotting configs and data processing
"""
# Read the different electrical parameters from the csv file, drop the NAN values and store it to a variable names plot_*
plot_current = cycle["Current, A"].dropna()
plot_cycle = cycle["Step"].dropna()
plot_voltage = cycle["Voltage, V"].dropna()
plot_temperature = cycle["Temperature A1, degC"].dropna()
plot_AH = cycle["Amp-Hours, AH"].dropna()
plot_realtime = cycle["Real Time"].dropna()
plot_mode = cycle["Mode"].dropna()
# As dataset is a 10 second interval data capture, create a variable with constant values from 0 to length of dataset with 10 seconds interval
csv_time_seconds = list(range(0,len(plot_current)))
plot_time = np.array(csv_time_seconds)*10
# Initialize a plot with 4 subplots in it
# Plot0 is operation mode vs time
# Plot1 is voltage vs time
# Plot2 is current vs time
# Plot3 is temperature vs time
# Plot4 is Battery AH capacity vs time
fig, ax = plt.subplots(6)
ax[0].plot(plot_time,plot_mode.values, 'tab:cyan')
ax[0].set_title('Battery operation mode',fontweight='bold')
ax[0].set(ylabel='Operation mode')
ax[1].plot(plot_time,plot_cycle.values, 'tab:cyan')
ax[1].set_title('Battery operation Step',fontweight='bold')
ax[1].set(ylabel='Operation Step')
ax[2].plot(plot_time,plot_voltage.values, 'tab:red')
ax[2].set_title('Battery voltage',fontweight='bold')
ax[2].set(ylabel='Voltage in V')
ax[3].plot(plot_time,plot_current.values, 'tab:green')
ax[3].set_title('Battery Current',fontweight='bold')
ax[3].set(ylabel='Current in A')
ax[4].plot(plot_time,plot_temperature.values, 'tab:blue')
ax[4].set_title('Battery Temperature',fontweight='bold')
ax[4].set(ylabel='Temperature in degC')
ax[5].plot(plot_time,plot_AH.values, 'tab:orange')
ax[5].set_title('Battery AH',fontweight='bold')
ax[5].set(xlabel='Time in seconds', ylabel='Capacity in AH')
plt.subplots_adjust(left=0.051,right=0.981,bottom=0.05,top=0.964,hspace=0.44) #This parameter is used to adjust the subplot parameters like spacing between plots, top spacing etc..
#plt.subplot_tool()  #This is the tool to adjust the graph parameters realtime and get the values, can also be used to see the default values
plt.show() #Command to show the plot in a different window