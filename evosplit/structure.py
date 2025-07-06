import numpy as np
import os
import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB import DSSP
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
class Structure:
    def __init__(self, pdb_file):
        self.pdb = pdb_file
        if self.pdb.endswith(".pdb"):
            p = PDBParser(PERMISSIVE=1)
        elif self.pdb.endswith(".cif"):
            p = MMCIFParser()
        self.structure = p.get_structure('tmp', self.pdb)
        num_models = len([m for m in self.structure.get_models()])
        if num_models == 0:
            raise ValueError("Empty file.")
        elif num_models > 1:
            warnings.warn("The number of models is more than 1.", Warning)
        self.model = self.structure[0]
        
    def get_deposition_date(self):
        try:
            return self.structure.header['deposition_date']
        except:
            return None
        
    def get_seq(self, chainid=None):
        """Obtaining protein sequences with coordinate information""" 
        id_to_resi = {}
        
        if chainid is None:
            chain = self.model
        else:
            chain = self.model[chainid]
        for resi in chain.get_residues():
            if resi.get_id()[1] in id_to_resi.keys():
                # 突变位置仅保留第一个残基
                continue
            if resi.get_id()[0].strip():
                continue
            elif [atom.get_name() for atom in resi] == ['CA']: 
                continue
            else:
                if resi.get_resname() in restype_3to1.keys():
                    id_to_resi[resi.get_id()[1]] = restype_3to1[resi.get_resname()]
        return "".join(id_to_resi.values()), list(id_to_resi.keys())
    
    def get_atoms(self, chainid=None):
        if chainid is None:
            chain = self.model
        else:
            chain = self.model[chainid]
        return [atom for atom in chain.get_atoms()]
        
    def pdb_to_dis(self, seq_mask=[]):
        return self.calc_dist_matrix(self.structure, seq_mask)

    def pdb_to_contact(self, seq_mask=[], cutoff=8):
        dist = self.calc_dist_matrix(self.structure, seq_mask)
        contact = np.zeros_like(dist)
        contact[np.where(dist<=cutoff)]=1
        for i in np.where(np.array(seq_mask)==0)[0]:
            contact[i, :] = 0
            contact[:, i] = 0
        return contact
        
    def calc_residue_dist(self, residue_one, residue_two) :
        """Returns the C-beta distance between two residues"""
        if residue_one.resname == "GLY":
            if 'CA' in residue_one.child_dict.keys():
                cb1 = residue_one["CA"]
            else:
                return 0
        else:
            if 'CB' in residue_one.child_dict.keys():
                cb1 = residue_one["CB"]
            else:
                return 0
        if residue_two.resname == "GLY":
            if 'CA' in residue_two.child_dict.keys():
                cb2 = residue_two["CA"]
            else:
                return 0
        else:
            if 'CB' in residue_two.child_dict.keys():
                cb2 = residue_two["CB"]
            else:
                return 0
        return cb1-cb2

    def calc_dist_matrix(self, structure: Bio.PDB.Structure.Structure, seq_mask=[]) :
        """Returns a matrix of C-beta distances map of a protein"""
        residues = [resi for resi in structure[0].get_residues() if resi.get_id()[0][0] != "H"]
        if seq_mask:
            length = len(seq_mask)
            resi_id = np.where(np.array(seq_mask)==1)[0]
        else:
            length = len(residues)
            resi_id = [i for i in range(length)]
        distances_map = np.zeros((length, length))
        for row, residue_one in zip(resi_id, residues):
            for col, residue_two in zip(resi_id, residues):
                distance = self.calc_residue_dist(residue_one, residue_two)
                distances_map[row, col] = distance
        return distances_map

    def get_bfactors(self, seq_mask=[]) :
        """Returns a list of Ca-plddt of a protein"""
        residues = [resi for resi in self.structure[0].get_residues() if resi.get_id()[0][0] != "H"]
        length = len(residues)
        bfactors = np.zeros(length)
        if seq_mask:
            resi_id = np.where(np.array(seq_mask)==1)[0]
        else:
            resi_id = [i for i in range(length)]
        for id, residue in zip(resi_id, residues):
            bfactors[id] = residue['CA'].get_bfactor()
        return bfactors
    
    def get_plddt(self, seq_mask=[]):
        bfactors = self.get_bfactors(seq_mask)
        return sum(bfactors)/len(bfactors)
    
    def get_SS8(self):
        dssp = DSSP(self.model, self.pdb)
        return {key[1][1]: dssp[key][2] for key in dssp.keys()}
        # return [dssp[key][2] for key in list(dssp.keys())]
    
    def get_SS3(self):
        SS8_SS3 = {'G':'H', 'H':'H', 'I':'H',
           'B':'E', 'E':'E',
           'S':'C', 'T':'C', '-':'C'}
        return {key: SS8_SS3[value] for key, value in self.get_SS8().items()}
        # return [SS8_SS3[key] for key in self.get_SS8()]
    
    def get_CA_coords(self, chain_id):
        coords = []
        for resi in self.model[chain_id]:
            coords.append(resi['CA'].get_coord())
        return np.stack(coords)
        
    # def save_partial_coords(self, chain_id, resi_ids, atom='all', output="./tmp.pdb"):
    #     new_model = Model(0)    
    #     new_chain = Chain(chain_id)
    #     if atom == 'all':
    #         for resi_id in resi_ids:
    #             new_chain.add(self.structure[0][chain_id][resi_id])
    #     else:
    #         for resi_id in resi_ids:
    #             residue = self.structure[0][chain_id][resi_id]
    #             new_residue = Residue(residue.id, residue.resname, residue.segid)
    #             for a in atom:
    #                 new_residue.add(residue[a])
    #             new_chain.add(new_residue)
    #     new_model.add(new_chain)
    #     io = PDBIO()
    #     io.set_structure(new_model)
    #     io.save(output)

    def save_partial_coords(self, chain_id, resi_ids, atom='all', output="./tmp.pdb"):
        new_model = Model(0)    
        new_chain = Chain(chain_id)
        seen_residues = set()  # 用于跟踪已处理的残基位置

        for resi_id in resi_ids:
            # 检查该位置是否已经处理过
            if resi_id in seen_residues:
                continue

            # 构造正确的残基ID格式
            full_id = (' ', resi_id, ' ')
            
            # 获取该位置的所有残基
            residues = [res for res in self.structure[0][chain_id] if res.get_id()[1] == resi_id]
            
            if not residues:
                print(f"Residue {resi_id} not found in chain {chain_id}. Skipping.")
                continue

            # 只使用第一个残基
            residue = residues[0]
            seen_residues.add(resi_id)

            if atom == 'all':
                new_chain.add(residue)
            else:
                new_residue = Residue(residue.id, residue.resname, residue.segid)
                for a in atom:
                    if a in residue:
                        new_residue.add(residue[a])
                new_chain.add(new_residue)

        new_model.add(new_chain)
        io = PDBIO()
        io.set_structure(new_model)
        io.save(output)
    
    def save_partial_chains(self, chains: list, output_dir="./tmp"):
        id_to_chains = {0:"A", 1:"B"}
        new_model = Model(0)
        for i, c in enumerate(chains):
            new_chain = Chain(id_to_chains[i])
            for resi in [i for i in self.model[c].get_residues()]:
                if resi.get_id()[0].strip():
                    continue
                else:
                    new_chain.add(resi)
            new_model.add(new_chain)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        io = PDBIO()
        io.set_structure(new_model)
        pdbid = self.pdb.split("/")[-1][:-4]
        path = f"{output_dir}/{pdbid}_{'_'.join(chains)}.pdb"
        io.save(path)
        return path

    def is_only_ca_atoms(self, chainid):
        for residue in self.model[chainid]:
            atom_names = [atom.get_name() for atom in residue]
            if atom_names == ['CA']: 
                return True
        return False
            
