import pandas as pd
import numpy as np
import pickle as pkl
import sklearn.model_selection as skl_mod_sel

# Load the alphabet
alphabet_path = "./zinc_alphabet.csv"
alphabet_dict = {}
with open(alphabet_path, "r") as file:
    counter = 0
    for line in file:
        alphabet_dict[line.strip()] = counter
        counter += 1
alph_size = len(alphabet_dict)

# Load the zinc data
zinc_path = "./chemical_vae/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
zinc_df = pd.read_csv(zinc_path)
zinc_df["smiles_hot"] = ""

# Create one-hot encoding
max_length = 110
char_freq = {}
for idx, row in zinc_df.iterrows():
    smiles_adj = row["smiles"].strip().strip('"')
    padding_coef = max_length-len(smiles_adj)
    smiles_padded = "&" + smiles_adj + "$" + "^"*padding_coef
    onehot_enc = np.zeros([alph_size, max_length+2])
    
    for jdx, char in enumerate(smiles_padded):
        onehot_enc[alphabet_dict[char], jdx] = 1
    zinc_df.at[idx, "smiles_hot"] = onehot_enc
    
    for char in smiles_padded:
        counter = char_freq.get(alphabet_dict[char], 0)
        char_freq[alphabet_dict[char]] = counter+1
    
min_val = min(char_freq.values())
for key, value in char_freq.items():
    char_freq[key] = min_val/value
char_weights = [value for key, value in sorted(char_freq.items())]


# Split into train and test sets
test_size, random_state = 0.1, 12121995
train_set, test_set = skl_mod_sel.train_test_split(zinc_df, shuffle=True, test_size=test_size, random_state=random_state)

# Pickle the results
with open("zinc_train_test.pkl", "wb") as file:
    pkl.dump((train_set, test_set, char_weights), file)

