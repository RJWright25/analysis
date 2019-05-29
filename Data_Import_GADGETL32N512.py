### RW_Init: basic presets

import matplotlib
import matplotlib.pyplot as plt
import warnings
import numpy as np
import h5py
import pickle
from RW_GenPythonTools import *
from RW_ParticleSTFTools import *
from RW_VRTools import *
warnings.filterwarnings("ignore")

# Data directories
run_directory="/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/snapshots/"
vr_directory="/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/VELOCIraptorG/"
tf_directory="/mnt/su3ctm/rwright/hydro_accretion/LB_L32N512/"
tf_treefile=tf_directory+"treesnaplist.txt"
particle_file_temp=run_directory+"snapshot_"+str(0).zfill(3)+".hdf5"
particle_file_temp=h5py.File(particle_file_temp)

#Simulation variables
sim_no_snaps=201
sim_snaps=[i for i in range(sim_no_snaps)] #will track halos from last snap here
sim_volume=32**3 #Mpc^3
sim_z0_snap=sim_snaps[-1]
sim_mdm=particle_file_temp['Header'].attrs['MassTable'][1]
sim_mgas=particle_file_temp['Header'].attrs['MassTable'][0]
sim_fb=sim_mgas/(sim_mdm+sim_mgas)

sim_timesteps=read_sim_timesteps(run_directory=run_directory,sim_type='GADGET',snap_no=201,files_lz=3)

######## halo_data keys ----

# 'ID'
# 'hostHaloID'
# 'numSubStruct'
# 'Mass_tot'
# 'Mass_200crit'
# 'M_gas'
# 'Xc'
# 'Yc'
# 'Zc'
# 'R_200crit'
# 'npart'
# 'SimulationInfo'
# 'UnitInfo'
# 'Head'
# 'Tail'
# 'HeadSnap'
# 'TailSnap'
# 'RootHead'
# 'RootTail'
# 'RootHeadSnap'
# 'RootTailSnap'
# 'HeadRank'
# 'Num_descen'
# 'Num_progen'
# 'Npart'
# 'Npart_unbound'
# 'Particle_IDs'
# 'Particle_Types'
# 'delta_m0_dt'
# 'delta_m1_dt'



# Import BASE halo data

if True:#if we want base halo data
    halo_data_base=read_vr_treefrog_data(vr_directory=vr_directory,vr_prefix="snapshot_",tf_name=tf_treefile,snap_no=sim_no_snaps,files_lz=3,files_type=2,files_nested=True,extra_halo_fields=['npart'],verbose=1)
    #once read, save this. 
    with open('halo_data_base.txt', 'wb') as halo_data_file:
        pickle.dump(halo_data_base, halo_data_file)
        halo_data_file.close()
            
else:
    with open('halo_data_base.txt', 'rb') as halo_data_file:
        halo_data_base=pickle.load(halo_data_file)
        halo_data_file.close()

# Append particle lists

if True:#if we want to make particle lists
    halo_data_all=add_particle_lists(vr_directory=vr_directory,vr_prefix="snapshot_",halo_data_all=halo_data_base,files_type=2,files_nested=True,files_lz=3,part_data_from_snap=180,verbose=1)
    with open('halo_data_appended.txt', 'wb') as halo_data_file:
        pickle.dump(halo_data_all, halo_data_file)
        halo_data_file.close()
else:#if we want to load particle lists
    with open('halo_data_appended.txt', 'rb') as halo_data_file:
        halo_data_all=pickle.load(halo_data_file)
        halo_data_file.close()




