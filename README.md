The publication "Ensembling methods for protein-ligand binding affinity prediction" describes a method (EBA) for predicting protein-ligand binding affinity using ensembles of deep learning models. Here's a breakdown of how to generate input feature sets and use them with a pre-trained ensemble model:

**1. Generating Input Feature Sets**

The EBA method uses five input features, which are combined in different ways to train individual models within the ensemble. To generate input features for new protein-ligand pairs, you must extract the following:

*   **Ligand Atoms (l):**
    *   Categorize each ligand atom into one of nine groups: B, C, N, O, P, S, Se, Halogen, and Metal. Encode this as a 9D one-hot vector.
    *   Determine eight atom properties: aromatic, ring, hydrogen bond donor, acceptor, partial charge, hybridization, heavy degree (number of heavy atoms connected), and hetero degree (number of hydrogen atoms connected).
        *   Encode aromatic, ring, hydrogen bond donor, and acceptor as binary vectors.
        *   Encode partial charge, hybridization, heavy degree, and hetero degree as real value vectors.
    *   Concatenate these eight properties into an 8D vector.
    *   Concatenate the 9D one-hot vector and the 8D vector to create a 17D feature vector for each atom.
    *   If a ligand has fewer than 50 heavy atoms, pad the matrix. The final representation is a matrix of size 50 x 17.

*   **Ligand SMILES (s):**
    *   Obtain the SMILES string for the ligand.
    *   Represent each character in the SMILES string with a unique integer from 1 to 64, based on a predefined alphabet.
    *   Encode each character using a 64D one-hot vector.
    *   If the SMILES string has fewer than 150 characters, pad the matrix. The final representation is a matrix of size 150 x 64.

*   **Angles (a):**
    *   Calculate the angle between a Cα atom of a residue in the protein, a ligand atom (C, N, O, or S), and the Cα atom of the consecutive residue.
    *   Discretize the angle values into 41 bins: 40 bins covering the range from 0° to 20° with an interval of 0.5°, and one bin for values larger than 20°.
    *   Flatten the 2D matrix into a 1D vector.

*   **Protein (p):**
    *   Represent each protein sequence as a 2D matrix.
    *   For each residue, determine:
        *   Amino acid type (21D one-hot vector, including an "unknown" type).
        *   Secondary structure element (SSE) from DSSP (8D one-hot vector for eight states: G, H, I, E, B, T, S, C).
        *   Physicochemical properties (11D one-hot vector for polar, non-polar, basic, acidic, and seven clusters based on dipoles and side chain volumes).
    *   Concatenate these three features to create a 40D feature vector for each residue.
    *   If the protein has fewer than 1000 residues, pad the matrix. The final representation is a matrix of size 1000 x 40.

*   **Pocket (t):**
    *   Represent the pocket region with the same features as the protein (amino acid sequence, SSEs, and physicochemical properties).
    *   Consider at most 63 residues in the pocket.
    *   The final representation is a matrix of size 63 x 40.

**2. Inputting Features into a Pre-trained Ensemble Model**

1.  **Feature Combinations:** The EBA method trains 13 individual models using different combinations of the five input features. The combinations are: `la`, `as`, `st`, `lt`, `lap`, `aps`, `lpt`, `pst`, `laps`, `lapt`, `lpst`, `apst`, and `lapst`. You need to generate the appropriate feature sets based on which model you are using.

2.  **Pre-trained Models:** The trained models and code are downloaded herein.

3.  **Model Input:** The models use embedding layers, cross-attention layers (in some models), CNN blocks, self-attention layers, and fully connected layers. The specific architecture depends on the feature combination used for each model.

4.  **Ensembling:**
    *   **Average Ensembling:**  Take the average of the predicted binding affinity values from each model in the ensemble.
    *   **FCNN-based Ensembling:** Use a Fully Connected Neural Network (FCNN) to combine the predictions from the individual models.

**Important Considerations:**

*   **Data Preprocessing:** Ensure that the input features are preprocessed in the same way as they were during the training of the EBA models. This includes normalization, padding, and any other transformations.
*   **Software and Libraries:** The models were implemented in PyTorch 1.13.0 with CUDA version 11.6. These libraries are already installed in the user's environment.