### RW_Init: basic presets

import matplotlib
import matplotlib.pyplot as plt
import warnings
import numpy as np
import h5py
import pickle
import splotch
from RW_GenPythonTools import *
from RW_ParticleSTFTools import *
from RW_VRTools import *
warnings.filterwarnings("ignore")

# Data directories
run_directory="/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/snapshots/"
vr_directory="/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/VELOCIraptorG/"
tf_directory="/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/TreeFrogG"
tf_treefile=tf_directory+"treesnaplist.txt"

#Simulation variables
sim_no_snaps=201
sim_snaps=[i for i in range(sim_no_snaps)] #will track halos from last snap here
sim_volume=25**3 #Mpc^3
sim_z0_snap=sim_snaps[-1]
sim_fb=0.120351/(0.120351+0.644648)

# Import BASE halo data + particle lists

sim_timesteps=read_sim_timesteps(run_directory=run_directory,sim_type='GADGET',snap_no=201)

# Add delta_m0 and delta_m1 to each halo -- different trimming, different snap smoothing
create_new=False
read_data=True

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


if read_data:
    if create_new:
        halo_data_base=read_vr_treefrog_data(vr_directory=vr_directory,snap_no=sim_no_snaps,extra_halo_fields=['npart'],halo_TEMPORALHALOIDVAL=1000000,verbose=1)
        with open('halo_data_base.txt', 'wb') as halo_data_file:
            pickle.dump(halo_data_base, halo_data_file)
            halo_data_file.close()

    else:
        with open('halo_data_base.txt', 'rb') as halo_data_file:
            halo_data_base = pickle.load(halo_data_file)
            halo_data_file.close()



    if create_new:
        halo_data=gen_delta_npart(halo_data=halo_data_base,sim_timesteps=sim_timesteps,snaps=list(range(121,200,1)),trim_hoes=True,depth=1)
        with open('halo_data_appended_trimmed_d1.txt', 'wb') as halo_data_file:
            pickle.dump(halo_data, halo_data_file)
            halo_data_file.close()

    else:
        with open('halo_data_appended_trimmed_d1.txt', 'rb') as halo_data_file:
            halo_data_appended_trimmed_d1 = pickle.load(halo_data_file)
            halo_data_file.close()

    if create_new:
        halo_data=gen_delta_npart(halo_data=halo_data_base,sim_timesteps=sim_timesteps,snaps=list(range(121,200,1)),trim_hoes=False,depth=1)
        with open('halo_data_appended_d1.txt', 'wb') as halo_data_file:
            pickle.dump(halo_data, halo_data_file)
            halo_data_file.close()

    else:
        with open('halo_data_appended_d1.txt', 'rb') as halo_data_file:
            halo_data_appended_d1 = pickle.load(halo_data_file)
            halo_data_file.close()

# halo_data_appended_trimmed_d1
# halo_data_appended_trimmed_d2
# halo_data_appended_trimmed_d3
# halo_data_appended_trimmed_d4
# halo_data_appended_trimmed_d5

# halo_data_appended_d1
# halo_data_appended_d2
# halo_data_appended_d3
# halo_data_appended_d4
# halo_data_appended_d5

hd_t=halo_data_appended_trimmed_d1
hd_nt=halo_data_appended_d1


