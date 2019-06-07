#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################

#*** Packages ***

if True:
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import pandas as pd
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM,z_at_value

    # VELOCIraptor python tools 
    from RW_VRTools import *

########################################################################################################################################################################
############################################################################## READ MASS DATA ##########################################################################
########################################################################################################################################################################

def read_mass_table(run_directory,sim_type='SWIFT',snap_prefix="snap_",snap_lz=4):

    #return mass of PartType0, PartType1 particles in sim units
    temp_file=h5py.File(run_directory+snap_prefix+str(0).zfill(snap_lz)+".hdf5")

    if sim_type=='SWIFT':
        M0=temp_file['PartType0']['Masses'][0]
        M1=temp_file['PartType1']['Masses'][1]
        return np.array([M0,M1])

    if sim_type=='GADGET':
        M0=temp_file['Header'].attrs['MassTable'][0]
        M1=temp_file['Header'].attrs['MassTable'][1]
        return np.array([M0,M1])

########################################################################################################################################################################
############################################################################## CREATE HALO DATA ########################################################################
########################################################################################################################################################################

def gen_halo_data_all(snaps=[],tf_treefile="listtreefiles.txt",vr_directory="",vr_prefix="snap_",vr_files_type=2,vr_files_nested=False,vr_files_lz=4,extra_halo_fields=[],halo_TEMPORALHALOIDVAL=[],verbose=1):
    # reads velociraptor and treefrog outputs with desired data fields (always includes ['ID','hostHaloID','numSubStruct','Mass_tot','Mass_200crit','M_gas','Xc','Yc','Zc','R_200crit'])

    ##### inputs
    # vr_directory: STRING for directory of VELOCIRAPTOR outputs
    # snap_no: INTEGER number of snapshots in simulation (needs ALL to create merger trees etc)
    # datafields: LIST of halo data fields from VR STF output (on top of the defaults)
    # snaps: LIST of INTEGER SNAPS
    # halo_TEMPORALHALOIDVAL: from VR (default halo_TEMPORALHALOIDVAL=1000000)

    ##### returns
    # list (for each snap) of dictionaries (each field) containing field data for each halo (AND concatenated particle lists for each halo)

    halo_data_all=[]
    halo_fields=['ID','hostHaloID','numSubStruct','Mass_tot','Mass_200crit','M_gas','Xc','Yc','Zc','R_200crit']#default halo fields
    halo_fields.extend(extra_halo_fields)
    
    if snaps==[]:#if no snaps specified, find them all!
        sim_snaps=list(range(1000))
        if verbose:
            print("Looking for snaps up to 1000")
    elif type(snaps)==list:
        sim_snaps=snaps
    elif type(snaps)==int:
        sim_snaps=list(range(snaps))

    print('Reading halo data using VR python tools')

    err=0
    found=0
    for isnap,snap in enumerate(sim_snaps):
        if verbose:
            print('Searching for halo data at snap = ',snap)

        if vr_files_nested:
            halo_data_snap=ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True)
        else:
            halo_data_snap=ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True)

        if not halo_data_snap==[]:
            halo_data_all.append(halo_data_snap)
            halo_data_all[isnap][0]['Snap']=snap
            found=found+1
        else:
            err=err+1
            if verbose:
                print("Couldn't find velociraptor files for snap = ",snap)
        
            if err>2 and found<2:#not finding files -- don't bother continuing
                print("Failed to find file on multiple occasions - terminating.")
                print("Check file pointer inputs.")
                return []

            if err>2 and found>1:#reached end of snaps
                print("Reached end of snapshots, total number of snaps found = ",len(halo_data_all))
                break

    # List of number of halos detected for each snap, List isolated data dictionary for each snap (in dictionaries)
    halo_data_counts=[item[1] for item in halo_data_all]
    halo_data_all=[item[0] for item in halo_data_all]
    snap_no=len(halo_data_all)
    sim_snaps=[halo_data_all[isnap]['Snap'] for isnap in range(snap_no)]

    # List sim info and unit info for each snap (in dictionaries)
    halo_siminfo=[halo_data_all[snap]['SimulationInfo'] for snap in sim_snaps]
    halo_unitinfo=[halo_data_all[snap]['UnitInfo'] for snap in sim_snaps]
    
    # import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)

    print('Now assembling descendent tree using VR python tools')

    halo_tree=ReadHaloMergerTreeDescendant(tf_treefile,ibinary=vr_files_type,iverbose=verbose+1,imerit=True,inpart=False)

    if halo_TEMPORALHALOIDVAL==[]:#if not given halo TEMPORALHALOIVAL, use the vr default
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose)
    else:
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose,TEMPORALHALOIDVAL=halo_TEMPORALHALOIDVAL)
    
    print('Finished assembling descendent tree using VR python tools')

    if verbose==1:
        print('Adding timesteps & filepath information')
    

    H0=halo_data_all[0]['SimulationInfo']['h_val']*halo_data_all[0]['SimulationInfo']['Hubble_unit']
    Om0=halo_data_all[0]['SimulationInfo']['Omega_Lambda']
    cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

    for isnap,snap in enumerate(sim_snaps):
        scale_factor=halo_data_all[isnap]['SimulationInfo']['ScaleFactor']
        redshift=z_at_value(cosmo.scale_factor,scale_factor,zmin=-0.5)
        lookback_time=cosmo.lookback_time(redshift).value

        halo_data_all[isnap]['SimulationInfo']['z']=redshift
        halo_data_all[isnap]['SimulationInfo']['LookbackTime']=lookback_time

        if vr_files_nested:
            halo_data_all[isnap]['FilePath']=vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz)
            halo_data_all[isnap]['FileType']=vr_files_type
        else:
            halo_data_all[isnap]['FilePath']=vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)
            halo_data_all[isnap]['FileType']=vr_files_type


    return halo_data_all

#########################################################################################################################################################################
##################################################################### GET PARTICLE LISTS ################################################################################
#########################################################################################################################################################################

def get_particle_lists(snap,halo_data_snap,add_subparts_to_fofs=False,verbose=1):
    
    if verbose:
        print('Reading particle lists for snap = ',snap)

    try:
        part_data_temp=ReadParticleDataFile(halo_data_snap['FilePath'],ibinary=halo_data_snap['FileType'],iverbose=0,iparttypes=1)
        
        if part_data_temp==[]:
            part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
            print('Particle data not found for snap = ',snap)
            print('Used directory: ',vr_directory+vr_prefix+str(snap).zfill(vr_files_lz))
            return part_data_temp


    except: #if we can't load particle data
        if verbose:
            print('Particle data not included in hdf5 file for snap = ',snap)
        part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
        return part_data_temp

    if add_subparts_to_fofs:

        if verbose==1:
            print('Appending FOF particle lists with substructure')
        
        field_halo_indices_temp=np.where(halo_data_snap['hostHaloID']==-1)[0]#find field/fof halos

        for i_field_halo,field_halo_ID in enumerate(halo_data_snap['ID'][field_halo_indices_temp]):#go through each field halo
            
            sub_halos_temp=(np.where(halo_data_snap['hostHaloID']==field_halo_ID)[0])#find the indices of its subhalos

            if len(sub_halos_temp)>0:#where there is substructure

                field_halo_temp_index=field_halo_indices_temp[i_field_halo]
                field_halo_plist=part_data_temp['Particle_IDs'][field_halo_temp_index]
                field_halo_tlist=part_data_temp['Particle_Types'][field_halo_temp_index]
                
                sub_halos_plist=np.concatenate([part_data_temp['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                sub_halos_tlist=np.concatenate([part_data_temp['Particle_Types'][isub] for isub in sub_halos_temp])#list all particles types substructure

                part_data_temp['Particle_IDs'][field_halo_temp_index]=np.concatenate([field_halo_plist,sub_halos_plist])#add particles to field halo particle list
                part_data_temp['Particle_Types'][field_halo_temp_index]=np.concatenate([field_halo_tlist,sub_halos_tlist])#add particles to field halo particle list
                part_data_temp['Npart'][field_halo_temp_index]=len(part_data_temp['Particle_IDs'][field_halo_temp_index])#update Npart for each field halo

        if verbose==1:
            print('Finished appending FOF particle lists with substructure')

    return part_data_temp

##########################################################################################################################################################################
############################################################################## CREATE PARTICLE HISTORIES #################################################################
##########################################################################################################################################################################

def gen_particle_history(snap,halo_data,verbose=0):
    ##### inputs
    # halo_data (from above)

    ##### returns
    # list (for each snap) of dictionaries (sub/field ids and types) of lists (for each bound particle) of:
    # (1) unique particle IDs of particles bound in field halos
    # (2) unique particle IDs of particles bound in subhalos

    sub_part_ids=[]
    all_part_ids=[]

    print('Generating particle histories up to snap = ',snap)
    running_list_all=[]
    running_list_sub=[]

    for isnap in range(snap):

        new_particle_data=get_particle_lists(snap=isnap,halo_data_snap=halo_data[isnap],add_subparts_to_fofs=False,verbose=verbose)
        
        if len(new_particle_data['Particle_IDs'])==0 or len(halo_data[isnap]['hostHaloID'])<2:#if no halos or no new particle data
            continue

        else:
            if verbose==True:
                print('Have particle lists for snap = ',isnap)
                        
            sub_halos_temp=(np.where(halo_data[isnap]['hostHaloID']>0)[0])#find the indices all subhalos

            if len(sub_halos_temp)>1:
                all_halos_plist=np.concatenate(new_particle_data['Particle_IDs'])
                sub_halos_plist=np.concatenate([new_particle_data['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                    
                running_list_all=np.concatenate([running_list_all,all_halos_plist])
                running_list_sub=np.concatenate([running_list_sub,sub_halos_plist])

                running_list_all=np.unique(running_list_all)
                running_list_sub=np.unique(running_list_sub)
                

    print('Unique particle histories created')
    return {'all_ids':running_list_all,'sub_ids':running_list_sub}

##########################################################################################################################################################################
############################################################################## CALC DELTA_N ##############################################################################
##########################################################################################################################################################################

def gen_accretion_rate(halo_data,snap,mass_table,particle_histories=[],depth=5,trim_particles=True,verbose=1): 

    ##### inputs
    # halo_data (from above - needs particle lists)
    # snaps: list of snaps to consider
    # trim_particles: get rid of particles which have been part of structure before when calculating accretion rate?

    ##### returns
    #list of accretion rates for each halo with key 'delta_mdm' and 'delta_mgas'

    sim_unit_to_Msun=halo_data[0]['UnitInfo']['Mass_unit_to_solarmass']
    m_0=mass_table[0]*sim_unit_to_Msun #MSol
    m_1=mass_table[1]*sim_unit_to_Msun #MSol
    print(m_0/m_1)

    if trim_particles:
        if particle_histories==[]:
            try:
                particle_history=gen_particle_history(halo_data=halo_data,snap=snap-depth-1)#obtain particles which have been part of substructure up to snap-depth-1
                allstructure_history=particle_history['all_ids']#obtain particles which have been part of any structure up to snap-depth-1
                substructure_history=particle_history['sub_ids']#obtain particles which have been part of sub structure up to snap-depth-1
            except:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1)
                return []
        else:
            substructure_history=particle_histories['sub_ids']
            allstructure_history=particle_histories['all_ids']

    def find_progen_index(index_0,snap,depth):
        id_0=halo_data[snap]['ID'][index_0]#the original id
        tail_id=halo_data[snap]['Tail'][index_0]#the tail id
        for idepth in range(1,depth+1,1):
            new_id=tail_id #the new id from tail in last snap
            if new_id in halo_data[snap-idepth]['ID']:
                new_index=np.where(halo_data[snap-idepth]['ID']==new_id)[0][0] #what index in the previous snap does the new_id correspond to
                tail_id=halo_data[snap-idepth]['Tail'][new_index] #the new id for next loop
            else:
                new_index=np.nan
                return new_index
             #new index at snap-depth
        return new_index

    isnap=-1
    isnap=isnap+1    

    if verbose:
        print('Generating accretion rates for snap = ',snap)

    #find final snap particle data
    part_data_2=get_particle_lists(snap,halo_data_snap=halo_data[snap],add_subparts_to_fofs=True,verbose=0)
    n_halos_2=len(part_data_2["Npart"])
    if n_halos_2==0:# if we can't find final particles or there are no halos
        print('Final particle lists not found at snap = ',snap)
        return []

    #find initial snap particle data
    part_data_1=get_particle_lists(snap-depth,halo_data_snap=halo_data[snap-depth],add_subparts_to_fofs=True,verbose=0)
    if snap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
        print('Initial particle lists not found at required depth (snap = ',snap-depth,')')
        return []

    delta_m0=[]
    delta_m1=[]

    for ihalo in range(n_halos_2):#for each halo
        progen_index=find_progen_index(index_0=ihalo,snap=snap,depth=depth)#find progen index
        if progen_index>-1:#if progen_index is valid
            part_IDs_init=part_data_1['Particle_IDs'][progen_index]
            part_IDs_final=part_data_2['Particle_IDs'][ihalo]
            part_Types_init=part_data_1['Particle_Types'][progen_index]
            part_Types_final=part_data_2['Particle_Types'][ihalo]
        else:
            delta_m0.append(np.nan)
            delta_m1.append(np.nan)
            continue

        new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))#list of particles new to halo
        new_particle_types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))#list of particle types new to halo
        
        ################# TRIMMING PARTICLES #################
        #get particle histories for the snap depth (minus 1)
        if trim_particles:
            if len(substructure_history)<100:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1)

            if halo_data[snap]['hostHaloID'][ihalo]==-1:#if a field halo
                for ipart,part_ID_temp in enumerate(new_particle_IDs):#for each particle new to this FIELD halo
                    if part_ID_temp in allstructure_history:#check if was ever in structure in the past
                        new_particle_types[ipart]=-1#if so, remove type. 

            else:#if a subhalo
                for ipart,part_ID_temp in enumerate(new_particle_IDs):#for each particle new to this SUB halo
                    if part_ID_temp in substructure_history or part_ID_temp not in allstructure_history:#check if was in substructure in past OR has never been part of structure
                        new_particle_types[ipart]=-1#if so, remove type. 

            new_particle_types = np.compress(np.logical_not(new_particle_IDs==-1),new_particle_types)

        ########### NOW WE HAVE THE DESIRED NEW (UNIQUE) PARTICLES FOR EACH HALO ###########
        delta_m0_temp=np.sum(new_particle_types==0)*m_0
        delta_m1_temp=np.sum(new_particle_types==1)*m_1
        delta_m0.append(delta_m0_temp)
        delta_m1.append(delta_m1_temp)
        ####################################################################################

    lt2=halo_data[snap]['SimulationInfo']['LookbackTime']
    lt1=halo_data[snap-depth]['SimulationInfo']['LookbackTime']
    delta_t=abs(lt1-lt2)#Gyr

    if mass_table[0]>mass_table[1]:#make sure m_dm is more massive (the more massive particle should be the dm particle)
        delta_m={'DM_Acc':np.array(delta_m0)/delta_t,'Gas_Acc':np.array(delta_m1)/delta_t,'dt':delta_t}
    else:
        delta_m={'DM_Acc':np.array(delta_m1)/delta_t,'Gas_Acc':np.array(delta_m0)/delta_t,'dt':delta_t}

    return delta_m