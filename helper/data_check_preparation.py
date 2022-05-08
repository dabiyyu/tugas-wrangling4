import pandas as pd
import numpy as np

def read_data(PATH):
    '''
    Read data from dataset from path
   
    Parameters
    ----------
    PATH : str
        path source of training data, csv.
    
    Returns
    -------
    data : pd.DataFrame
        Data for modeling
    '''
    data = pd.read_csv(PATH, encoding='latin-1')
    
    return data

def check_and_set_columns(data_input, COLUMN):
    '''
    Check data input consistency with predefined COLUMN
    Set data input columns as COLUMN
    
    Parameters
    ----------
    data_input: pd.DataFrame
        DataFrame for modeling
    COLUMN: set
        A set of columns which will be used for modeling
        
    Returns
    -------
    data_input: pd.DataFrame
        Checked dataset for columns consistency
    '''
    COLUMN = set(COLUMN.keys())
    columns_in_data = set(data_input.columns)
    
    if not COLUMN.issubset(columns_in_data):
        with open("warning_msg.txt", "a") as writer:
            writer.write("There is at least one column not in the data")
        raise ValueError("There is at least one column not in the data")
        
    data_input = data_input[list(COLUMN)]
    return data_input

def set_dtypes(data_input, KOLOM):
    '''
    Check data input datatypes consistency with predefined DTYPES
    Set data datatypes as DTYPE
    
    Parameters
    ----------
    data_input: pd.DataFrame
        DaraFrame for modeling
    
    Returns
    -------
    data: pd.DataFrame
        Checked dataset for columns consistency
    '''
    data = data_input.astype(KOLOM)
    return data

def check_read_data_success(data_input):
    '''
    Sanity check for data success
    
    Parameters
    ----------
    data
    
    '''
    if not data_input.notnull().sum().sum() > 0:
        with open("warning_msg.txt", "a") as writer:
            writer.write("You have missing values in full in at least one column")
            
    return data_input

def read_and_check_data(path, column):
    """Read and checking data."""
    print("start import data")
    df = read_data(path)
    print("done import data")
    print("start checking and set columns")
    df = check_and_set_columns(df, column)
    print("done checking and set columns")
    print("start set dtypes")
    df = set_dtypes(df, column)
    print("done set dtypes")
    print("start checking missing Passenger_Id")
    df = check_read_data_success(df)
    print("done checking read data success")
    
    return df