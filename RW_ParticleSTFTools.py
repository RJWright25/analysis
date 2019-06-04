#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################

#*** Packages ***

if True:
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM,z_at_value

    # VELOCIraptor python tools 
    from RW_VRTools import *


def read_mass_table(run_directory,sim_type='SWIFT',snap_prefix="snap_",snap_lz=4):

    #return mass of PartType0, PartType1 particles in sim units
    temp_file=h5py.File(run_directory+snap_prefix+str(0).zfill(snap_lz)+".hdf5")

    if sim_type=='SWIFT':
        M0=temp_file['PartType0']['Masses'][0]
        M1=temp_file['PartType1']['Masses'][0]
        return np.array([M0,M1])

    if sim_type=='GADGET':
        M0=temp_file['Header']['MassTable'][0]
        M1=temp_file['Header']['MassTable'][1]
        return np.array([M0,M1])


##########################################################################################################################################################################
############################################################################## CREATE HALO DATA ##########################################################################
##########################################################################################################################################################################

def read_vr_treefrog_data(vr_directory,vr_prefix,tf_treefile,vr_files_type=2,vr_files_nested=False,vr_files_lz=4,snap_no=200,extra_halo_fields=[],halo_TEMPORALHALOIDVAL=[],verbose=1):
    # reads velociraptor and treefrog outputs with desired data fields (always includes ['ID','hostHaloID','numSubStruct','Mass_tot','Mass_200crit','M_gas','Xc','Yc','Zc','R_200crit'])

    ##### inputs
    # vr_directory: STRING for directory of VELOCIRAPTOR outputs
    # snap_no: INTEGER number of snapshots in simulation (needs ALL to create merger trees etc)
    # datafields: LIST of halo data fields from VR STF output (on top of the defaults)
    # snaps: LIST of INTEGER SNAPS
    # halo_TEMPORALHALOIDVAL: from VR (default halo_TEMPORALHALOIDVAL=1000000)

    ##### returns
    # list (for each snap) of dictionaries (each field) containing field data for each halo (AND concatenated particle lists for each halo)

    sim_snaps=[i for i in range(snap_no)]
    halo_fields=['ID','hostHaloID','numSubStruct','Mass_tot','Mass_200crit','M_gas','Xc','Yc','Zc','R_200crit']
    halo_fields.extend(extra_halo_fields)
            
    if verbose==1:
        print('Reading halo data using VR python tools')

    # Load data from all desired snaps into list structure
    if vr_files_nested==False:
        halo_data_all=[ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True) for snap in sim_snaps]
    else:
        halo_data_all=[ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True) for snap in sim_snaps]

    if verbose==1:
        print('Finished reading halo data')

    # List of number of halos detected for each snap, List isolated data dictionary for each snap (in dictionaries)
    halo_data_counts=[item[1] for item in halo_data_all]
    halo_data_all=[item[0] for item in halo_data_all]

    # List sim info and unit info for each snap (in dictionaries)
    halo_siminfo=[halo_data_all[snap]['SimulationInfo'] for snap in sim_snaps]
    halo_unitinfo=[halo_data_all[snap]['UnitInfo'] for snap in sim_snaps]
    
    # import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)
    if verbose==1:
        print('Assembling descendent tree using VR python tools')

    halo_tree=ReadHaloMergerTreeDescendant(tf_treefile,ibinary=vr_files_type,iverbose=1,imerit=True,inpart=False)
    if halo_TEMPORALHALOIDVAL==[]:
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=0)
    else:
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=0,TEMPORALHALOIDVAL=halo_TEMPORALHALOIDVAL)
    
    if verbose==1:
        print('Finished assembling descendent tree using VR python tools')



    if verbose==1:
        print('Adding timestep information & finishing up')

    H0=halo_data_all[0]['SimulationInfo']['h_val']*halo_data_all[0]['SimulationInfo']['Hubble_unit']
    Om0=halo_data_all[0]['SimulationInfo']['Omega_Lambda']
    cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

    for snap in sim_snaps:
        scale_factor=halo_data_all[snap]['SimulationInfo']['ScaleFactor']
        redshift=z_at_value(cosmo.scale_factor,scale_factor,zmin=-0.5)
        lookback_time=cosmo.lookback_time(redshift).value

        halo_data_all[snap]['SimulationInfo']['z']=redshift
        halo_data_all[snap]['SimulationInfo']['LookbackTime']=lookback_time


    return halo_data_all

def gen_particle_lists(snap,halo_data_snap,vr_directory,vr_prefix,vr_files_type=2,vr_files_nested=False,vr_files_lz=4,verbose=1):
    
    if verbose==True:
        print('Reading particle lists for snap = ',snap)

    try:
        if vr_files_nested==False:
            part_data_temp=ReadParticleDataFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iverbose=0,iparttypes=1)
            if verbose:
                print('Particle data found for snap = ',snap)
        else:
            part_data_temp=ReadParticleDataFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iverbose=0,iparttypes=1)
            if verbose:
                print('Particle data found for snap = ',snap)

    except:#if we can't load particle data
        if verbose:
            print('Particle data not found for snap = ',snap)
        part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
        return part_data_temp

    if verbose==1:
        print('Appending FOF particle lists with substructure for snap = ',snap)

    field_halo_indices_temp=np.where(halo_data_snap['hostHaloID']==-1)[0]#find field/fof halos
    if len(field_halo_indices_temp)>0:#where there are field halos
        for field_halo_ID in halo_data_snap['ID'][field_halo_indices_temp]:#go through each field halo

            sub_halos_temp=(np.where(halo_data_snap['hostHaloID']==field_halo_ID)[0])#find its subhalos

            if len(sub_halos_temp)>0:#where there is substructure

                field_halo_temp_index=np.where(halo_data_snap['ID']==field_halo_ID)[0][0]
                field_halo_plist=part_data_temp['Particle_IDs'][field_halo_temp_index]
                field_halo_tlist=part_data_temp['Particle_Types'][field_halo_temp_index]
                sub_halos_plist=np.concatenate([part_data_temp['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                sub_halos_tlist=np.concatenate([part_data_temp['Particle_Types'][isub] for isub in sub_halos_temp])#list all particles types substructure

                part_data_temp['Particle_IDs'][field_halo_temp_index]=np.concatenate([field_halo_plist,sub_halos_plist])#add particles to field halo particle list
                part_data_temp['Particle_Types'][field_halo_temp_index]=np.concatenate([field_halo_tlist,sub_halos_tlist])#add particles to field halo particle list
                part_data_temp['Npart'][field_halo_temp_index]=len(part_data_temp['Particle_IDs'][field_halo_temp_index])#update Npart for each field halo

    if verbose==1:
        print('Finished with particle lists for snap = ',snap)

    return part_data_temp

##########################################################################################################################################################################
############################################################################## CREATE PARTICLE HISTORIES #################################################################
##########################################################################################################################################################################

def gen_part_history(halo_data,vr_directory,vr_prefix,vr_files_type=2,vr_files_nested=False,vr_files_lz=4,verbose=1):
    ##### inputs
    # halo_data (from above)

    ##### returns
    # list (for each snap) of dictionaries (sub/field ids and types) of lists (for each bound particle) of:
    # (1) unique particle IDs of particles bound in field halos
    # (2) unique particle IDs of particles bound in subhalos

    sub_part_ids=[[] for i in range(len(halo_data))]
    field_part_ids=[[] for i in range(len(halo_data))]

    for snap in range(len(halo_data)):

        particle_data_temp=gen_particle_lists(snap=snap,halo_data_snap=halo_data[snap],vr_directory=vr_directory,vr_prefix=vr_prefix,vr_files_type=vr_files_type,vr_files_nested=vr_files_nested)

        if len(particle_data_temp['Particle_IDs'])==0:#if no halos or no particle data
            field_part_ids[snap]=[]
            sub_part_ids[snap]=[]

        else:
            if verbose==True:
                print('Generating particle histories for snap = ',snap)

            field_halo_indices=np.where(halo_data[snap]['hostHaloID']==-1)[0]
            sub_halo_indices=np.where(halo_data[snap]['hostHaloID']>0)[0]

            #list of particles in field/subhalos for this snap
            field_part_ids_temp=[particle_data_temp['Particle_IDs'][field_halo_index] for field_halo_index in field_halo_indices]
            sub_part_ids_temp=[particle_data_temp['Particle_IDs'][sub_halo_index] for sub_halo_index in sub_halo_indices]

            field_part_ids_flattened=[]
            sub_part_ids_flattened=[]

            for halo_idlist_temp in field_part_ids_temp:
                field_part_ids_flattened.extend(halo_idlist_temp)

            for halo_idlist_temp in sub_part_ids_temp:
                sub_part_ids_flattened.extend(halo_idlist_temp)

            field_part_ids[snap]=np.concatenate((field_part_ids[snap-1],field_part_ids_flattened),axis=0) #list of particles in field halos at snap
            sub_part_ids[snap]=np.concatenate((sub_part_ids[snap-1],sub_part_ids_flattened),axis=0)#list of particles in subhalos at snap

    sub_part_ids_unique=[[] for i in range(len(halo_data))]
    field_part_ids_unique=[[] for i in range(len(halo_data))]

    for snap in range(len(halo_data)):
        sub_part_ids_unique[snap]=np.unique(sub_part_ids[snap])
        field_part_ids_unique[snap]=np.unique(field_part_ids[snap])

    print('Unique particle histories created')
    return [{"field_ids":field_part_ids_unique[snap],"sub_ids":sub_part_ids_unique[snap]} for snap in range(len(halo_data))]

##########################################################################################################################################################################
############################################################################## CALC DELTA_N ##############################################################################
##########################################################################################################################################################################

def gen_delta_npart(halo_data,snaps,unique_particle_list,mass_table,vr_directory,vr_prefix,vr_files_type=2,vr_files_nested=False,vr_files_lz=4,depth=5,trim_hoes=True,verbose=True): 

    ##### inputs
    # halo_data (from above - needs particle lists)
    # snaps: list of snaps to consider
    # trim_hoes: get rid of particles which have been part of structure before when calculating accretion rate?

    ##### returns
    #list of accretion rates for each halo with key 'delta_mdm' and 'delta_mgas'

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


    acc_dm=[[] for snap in snaps]
    acc_gas=[[] for snap in snaps]

    # iterate through each snap
    isnap=-1

    for snap in snaps:
        isnap=isnap+1

        if verbose==True:
            print('Generating accretion rates for snap = ',snap)
            print(isnap/len(snaps)*100,' \% done with snaps')

        #find final snap particle data
        part_data_2=gen_particle_lists(snap,halo_data_snap=halo_data[snap],vr_directory=vr_directory,vr_prefix=vr_prefix,vr_files_type=2,vr_files_nested=False,vr_files_lz=4,verbose=1)
        if part_data_2["Npart"]==[]:# if we can't find final particles
            print('Final particle lists not found at snap = ',snap)
            for ihalo in range(n_halo_2):
                halo_tracked=False
                part_IDs_2[ihalo]=[]
                part_Types_2[ihalo]=[]
            continue#go to next snap
        part_IDs_2=part_data_2['Particle_IDs']
        part_Types_2=part_data_2['Particle_Types']
        n_halo_2=len(halo_data[snap]['ID'])
        halo_tracked=np.zeros(n_halo_2)

        #find initial snap particle data
        part_data_1=gen_particle_lists(snap-depth,halo_data_snap=halo_data[snap-depth],vr_directory=vr_directory,vr_prefix=vr_prefix,vr_files_type=2,vr_files_nested=False,vr_files_lz=4,verbose=1)
        if snap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
            print('Initial particle lists not found at required depth (snap = ',snap-depth,')')
            for ihalo in range(n_halo_2):
                halo_tracked=False
                part_IDs_2[ihalo]=[]
                part_Types_2[ihalo]=[]
            continue#go to next snap

        #if we have found both initial and final particle lists...
        for ihalo in range(n_halo_2):
            progen_index=find_progen_index(index_0=ihalo,snap=snap,depth=depth)
            if progen_index>-1:
                part_IDs_1[ihalo]=part_data_1['Particle_IDs'][progen_index]
                part_Types_1[ihalo]=part_data_1['Particle_Types'][progen_index]
                halo_tracked[ihalo]=True
            else:
                part_IDs_1[ihalo]=[]
                part_Types_1[ihalo]=[]
                halo_tracked[ihalo]=False
                continue

        #now we finally have part_IDs_1 and part_IDs_2!
        #now find accretion rates for this snap
        delta_m0=[np.nan for i in range(n_halo_2)]
        delta_m1=[np.nan for i in range(n_halo_2)]
        
        #recall particle histories for the snap depth
        if trim_hoes:
            try:
                substructure_partID_list_uptosnap=unique_particle_list[snap-depth-1]["sub_ids"]
                structure_partID_list_uptosnap=unique_particle_list[snap-depth-1]["field_ids"]
            except:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1)
                continue



        for ihalo in range(n_halo_2):#for each halo, find accretion rate

            field_halo[ihalo]=halo_data[snap]['hostHaloID'][ihalo]==-1#True if dealing with field, False if dealing with Sub
            part_IDs_init=part_IDs_1[ihalo]
            part_IDs_final=part_IDs_2[ihalo]
            part_Types_init=part_Types_1[ihalo]
            part_Types_final=part_Types_2[ihalo]

            new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))
            new_particle_types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))
            
            ################# TRIMMING PARTICLES #################

            if trim_hoes==True:#trim particles which have been part of substructure

                if field_halo[ihalo]:#if a field halo
                    for ipart in range(len(new_particle_IDs)):#for each new particle
                        part_ID_temp=new_particle_IDs[ipart]
                        if part_ID_temp in structure_partID_list_uptosnap:#check if was ever in structure in the past
                            new_particle_IDs[ipart]=-1#if so, remove ID. 
                            new_particle_types[ipart]=-1

                else:#if a subhalo
                    host_ID=halo_data[snap]['hostHaloID'][ihalo]
                    for ipart in range(len(new_particle_IDs)):#for each new particle
                        part_ID_temp=new_particle_IDs[ipart]
                        if part_ID_temp in substructure_partID_list_uptosnap:#check if was ever in substructure in the past
                            new_particle_IDs[ipart]=-1#if so, remove ID. 
                            new_particle_types[ipart]=-1

                new_particle_IDs = np.compress(~(new_particle_IDs==-1),new_particle_IDs)
                new_particle_types = np.compress(~(new_particle_IDs==-1),new_particle_types)


            if halo_tracked[ihalo]:
                delta_n_tot[ihalo]=len(new_particle_IDs)
                
                ##### CHANGE HERE IF WE GET MORE PARTICLE TYPES
                sim_unit_to_Msun=halo_data[0]['UnitInfo']['Mass_unit_to_solarmass']
                m_0=mass_table[1]*sim_unit_to_Msun #MSol
                m_1=mass_table[0]*sim_unit_to_Msun #MSol

                delta_m0[ihalo]=np.sum(new_particle_types==0)*m_0
                delta_m1[ihalo]=np.sum(new_particle_types==1)*m_1

                
        lt2=halo_data[snap]['SimulationInfo']['LookbackTime']
        lt1=halo_data[snap-depth]['SimulationInfo']['LookbackTime']
    
        delta_t=abs(lt1-lt2)#Gyr

        if mass_table[0]>mass_table[1]:
            acc_dm[isnap]=delta_m0/delta_t
            acc_gas[isnap]=delta_m1/delta_t
        else:
            acc_dm[isnap]=delta_m1/delta_t
            acc_gas[isnap]=delta_m0/delta_t

    return acc_dm,acc_gas


