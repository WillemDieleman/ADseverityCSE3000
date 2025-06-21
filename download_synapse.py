import synapseclient
import synapseutils

# ROSMAP

syn_id_spareProcessed = "syn23554292"
syn_id_brainCellsMetadata = "syn23554294"
syn_id_brainGenesMetadata = "syn23554293"

syn_id_microglia = "syn53693904"
syn_id_vasc = "syn53693877"

syn_id_codebook = "syn3191090"
syn_id_metadata = "syn3191087"

if __name__ == "__main__":

    with open("token.txt", "r") as f:  # NOTE: Make sure to add input the token to this file when reproducing
        token = f.read().strip()
    syn = synapseclient.login(silent=True, authToken=token)


    # first download the regular rosmap data
    for syn_id in [
        syn_id_microglia
    ]:
        syn.get(syn_id, downloadLocation="data/ROSMAP", ifcollision="keep.local")