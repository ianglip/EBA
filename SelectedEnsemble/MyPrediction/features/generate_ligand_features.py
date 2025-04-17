\
import pickle
import argparse
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("RDKit is not installed. Please install it, e.g., using conda:")
    print("conda install -c conda-forge rdkit")
    exit()

# --- Constants based on the publication description ---
MAX_LIGAND_ATOMS = 50  # Maximum number of heavy atoms to consider
FEATURE_DIM = 17       # 9 (type) + 8 (properties)

# Define atom types for one-hot encoding
ATOM_TYPES = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'Halogen', 'Metal']
HALOGENS = ['F', 'Cl', 'Br', 'I']
METALS = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Mo', 'Al', 'Ga', 'Sn', 'Pb'] # Add more if needed

def get_atom_type_one_hot(atom):
    """Generates the 9D one-hot vector for atom type."""
    one_hot = np.zeros(9)
    symbol = atom.GetSymbol()
    if symbol in ATOM_TYPES:
        if symbol in HALOGENS:
            one_hot[ATOM_TYPES.index('Halogen')] = 1
        elif symbol in METALS:
             one_hot[ATOM_TYPES.index('Metal')] = 1
        else:
            one_hot[ATOM_TYPES.index(symbol)] = 1
    else:
        # Handle unknown atom types if necessary, maybe map to 'C' or raise error
        print(f"Warning: Unknown atom type '{symbol}'. Treating as Carbon.")
        one_hot[ATOM_TYPES.index('C')] = 1 # Default or placeholder
    return one_hot

def get_atom_properties(atom):
    """Calculates the 8D property vector for an atom."""
    properties = np.zeros(8)
    # Binary properties
    properties[0] = 1 if atom.GetIsAromatic() else 0
    properties[1] = 1 if atom.IsInRing() else 0
    properties[2] = 1 if atom.GetNumImplicitHs() > 0 or any(n.GetSymbol() == 'H' for n in atom.GetNeighbors() if n.GetAtomicNum()==1 and atom.GetAtomicNum() in [7, 8]) else 0 # Simple H-bond donor check (N/O attached to H) - RDKit features might be better
    properties[3] = 1 if atom.GetAtomicNum() in [7, 8] else 0 # Simple H-bond acceptor check (N or O) - RDKit features might be better

    # Real value properties
    # Partial Charge (requires calculation, Gasteiger charges are common)
    # Note: Calculating charges requires the whole molecule context.
    # We'll calculate it once per molecule and access it here.
    try:
        properties[4] = float(atom.GetProp('_GasteigerCharge'))
    except KeyError:
        print("Warning: Gasteiger charges not found. Setting partial charge to 0.")
        print("Ensure Chem.AllChem.ComputeGasteigerCharges(mol) was called.")
        properties[4] = 0.0 # Placeholder if charges weren't computed

    # Hybridization (encoded numerically)
    hybridization = atom.GetHybridization()
    if hybridization == Chem.rdchem.HybridizationType.SP:
        properties[5] = 1
    elif hybridization == Chem.rdchem.HybridizationType.SP2:
        properties[5] = 2
    elif hybridization == Chem.rdchem.HybridizationType.SP3:
        properties[5] = 3
    elif hybridization == Chem.rdchem.HybridizationType.SP3D:
        properties[5] = 4 # Or handle as needed
    elif hybridization == Chem.rdchem.HybridizationType.SP3D2:
        properties[5] = 5 # Or handle as needed
    else: # S, UNKNOWN, OTHER
        properties[5] = 0 # Default/unknown

    properties[6] = atom.GetDegree() # Number of bonded neighbors (excluding H)
    properties[7] = atom.GetTotalNumHs(includeNeighbors=True) # Number of attached hydrogens

    # Normalize or scale real values if necessary (based on training data)
    # For now, using raw values. Check if training used specific scaling.
    # Example scaling (if needed):
    # properties[4] = properties[4] / 10.0 # Scale charge
    # properties[6] = properties[6] / 5.0  # Scale heavy degree
    # properties[7] = properties[7] / 4.0  # Scale hetero degree

    return properties

def generate_ligand_features(ligand_file: Path, output_dir: Path, ligand_id: str):
    """
    Generates the 50x17 feature matrix for a ligand and saves it as a .pkl file.

    Args:
        ligand_file: Path to the input ligand file (e.g., SDF, MOL2).
        output_dir: Directory to save the output .pkl file.
        ligand_id: The identifier (e.g., 'mypdb') to use for the output filename.
    """
    print(f"Processing ligand: {ligand_id} from file: {ligand_file}")

    # --- Placeholder for User Input ---
    # Ensure the ligand_file path points to your actual ligand structure file.
    # Supported formats depend on RDKit (SDF, MOL, MOL2 are common).
    if not ligand_file.exists():
        print(f"Error: Ligand file not found at {ligand_file}")
        return

    # Load the molecule
    # Use a supplier for SDF/MOL2 which might contain multiple molecules
    # Assuming the first molecule in the file is the one we want
    mol = None
    if ligand_file.suffix.lower() == '.sdf':
        supplier = Chem.SDMolSupplier(str(ligand_file))
        if supplier and len(supplier) > 0:
            mol = supplier[0]
    elif ligand_file.suffix.lower() in ['.mol2', '.mol']:
         mol = Chem.MolFromMolFile(str(ligand_file)) # Or MolFromMol2File if needed
    else:
        print(f"Error: Unsupported file format {ligand_file.suffix}. Please use SDF, MOL, or MOL2.")
        return

    if mol is None:
        print(f"Error: Could not load molecule from {ligand_file}")
        return

    # Pre-calculate Gasteiger charges for the whole molecule
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"Warning: Could not compute Gasteiger charges: {e}. Partial charges will be 0.")


    # --- Feature Extraction ---
    atom_features_list = []
    heavy_atom_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1: # Only consider heavy atoms for the count/list
            if heavy_atom_count < MAX_LIGAND_ATOMS:
                type_vec = get_atom_type_one_hot(atom)
                prop_vec = get_atom_properties(atom)
                feature_vec = np.concatenate((type_vec, prop_vec))
                if feature_vec.shape[0] != FEATURE_DIM:
                     print(f"Error: Feature vector for atom {atom.GetIdx()} has wrong dimension: {feature_vec.shape[0]}. Expected {FEATURE_DIM}")
                     # Handle error appropriately
                     continue # Skip this atom or raise an error
                atom_features_list.append(feature_vec)
            heavy_atom_count += 1

    if heavy_atom_count > MAX_LIGAND_ATOMS:
        print(f"Warning: Ligand {ligand_id} has {heavy_atom_count} heavy atoms, exceeding MAX_LIGAND_ATOMS ({MAX_LIGAND_ATOMS}). Truncating features.")

    # --- Padding / Conversion to NumPy Array ---
    num_features = len(atom_features_list)
    feature_matrix = np.zeros((MAX_LIGAND_ATOMS, FEATURE_DIM), dtype=np.float32)

    if num_features > 0:
        feature_matrix[:num_features, :] = np.array(atom_features_list, dtype=np.float32)
    else:
        print(f"Warning: No heavy atoms found or processed for {ligand_id}.")


    # --- Save Output ---
    output_file = output_dir / f"{ligand_id}.pkl"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    print(f"Saving features for {ligand_id} to {output_file} with shape {feature_matrix.shape}")
    with open(output_file, "wb") as f:
        pickle.dump(feature_matrix, f)

    print("Ligand feature generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ligand Atom Features (Type 'l')")
    parser.add_argument("--ligand_file", type=Path, required=True,
                        help="Path to the input ligand file (e.g., myligand.sdf)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save the output .pkl feature file")
    parser.add_argument("--ligand_id", type=str, required=True,
                        help="Unique ID for the ligand (e.g., 'mypdb'), used for the output filename (*.pkl)")

    args = parser.parse_args()

    generate_ligand_features(args.ligand_file, args.output_dir, args.ligand_id)

# --- Example Usage ---
# python scripts/generate_ligand_features.py --ligand_file path/to/your/ligand.sdf --output_dir SelectedEnsemble/MyPrediction/features/ligand_atoms --ligand_id mypdb
