import pandas as pd
import numpy as np
import requests
import re



def download_aaindex(url):
    """
    Downloads the AAindex data from the specified URL.
    
    Parameters:
    url (str): The URL to download the data from.

    Returns:
    str: The raw text data from the URL.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.text


def parse_aaindex1(data):
    """
    Parses the AAindex1 data to extract the residue features.
    
    Parameters:
    data (str): The raw text data from the AAindex1 file.

    Returns:
    np.ndarray: A 20xF numpy array with the parsed features.
    """
    features_list = []
    entries = data.split('//\n')
    
    for entry in entries:
        if entry.strip():
            lines = entry.strip().split('\n')
            if len(lines) >= 2:
                feature_values = []
                # Take the last two lines of the entry
                values_lines = lines[-2:] #this is hardcoded, because currently the aaindex 
                #information is the two lines before the //
                
                for line in values_lines:
                    # Handle potential 'NA' values and skip them
                    values = line.split()
                    for value in values:
                        if value != 'NA':
                            feature_values.append(float(value))
                if len(feature_values) == 20:
                    features_list.append(feature_values)
    print(len(features_list))                
    features_array = np.array(features_list).T  # Transpose to get a 20xF array
    return features_array

def parse_aaindex3(data):
    """
    Parses the AAindex3 data to extract the pairwise features.
    
    Parameters:
    data (str): The raw text data from the AAindex3 file.

    Returns:
    np.ndarray: A 20x20xH numpy array with the parsed features.
    """
    features_list = []
    entries = data.split('//\n')
    
    for entry in entries:
        if entry.strip():
            lines = entry.strip().split('\n')
            if len(lines) >= 20:
                # Take the 20 lines above the entry
                matrix_lines = lines[-20:] #this is also hard-coded since the current
                #aaindex3 database has 20 lines above the // encoding the pairwise features
                
                # Check if it's a lower triangular matrix or a complete matrix
                is_lower_triangular = all(len(line.split()) == i + 1 for i, line in enumerate(matrix_lines))

                if not is_lower_triangular:
                    # Complete matrix
                    matrix = []
                    for line in matrix_lines:
                        row = [float(value) if value != 'NA' else np.nan for value in line.split()]
                        print(row)
                        matrix.append(row)
                    matrix = np.array(matrix)
                    
                else:
                    # Lower triangular matrix
                    matrix = np.empty((20, 20)) * np.nan
                    idx = 0
                    for i in range(20):
                        row_values = matrix_lines[idx].split()
                        for j in range(i + 1):
                            if row_values[j] != 'NA':
                                matrix[i, j] = float(row_values[j])
                                matrix[j, i] = float(row_values[j])
                        idx += 1
                if np.isnan(matrix).any():
                    continue
                else:
                    features_list.append(matrix)
    
    features_array = np.array(features_list)  # Shape: (H, 20, 20)
    features_array = features_array.transpose((1, 2, 0))  # Change shape to (20, 20, H)
    return features_array
def main():
    data_path = "/home/az2798/IDR_cons/data/"

    url_aaindex1 = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"

    # Download and parse the data for aaindex1
    aaindex1_data = download_aaindex(url_aaindex1)
    parsed_features_array = parse_aaindex1(aaindex1_data)
    np.save(f'{data_path}aa_index1.npy', parsed_features_array)


    url_aaindex3 = "https://www.genome.jp/ftp/db/community/aaindex/aaindex3"
    # Download and parse the data for aaindex3
    aaindex3_data = download_aaindex(url_aaindex3)
    parsed_features_array = parse_aaindex3(aaindex3_data)
    print(parsed_features_array.shape, "Pairwise Features shape")
    #print(parsed_features_array, "Pairwise Features")
    np.save(f'{data_path}aa_index3.npy', parsed_features_array)
if __name__ == "__main__":
    main()
