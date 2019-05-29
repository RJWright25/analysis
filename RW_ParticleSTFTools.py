#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################

#*** Packages ***

if True:
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from astropy.cosmology import FlatLambdaCDM

    # VELOCIraptor python tools 
    from RW_VRTools import ReadPropertyFile
    from RW_VRTools import TraceMainProgen
    from RW_VRTools import ReadHaloMergerTree
    from RW_VRTools import ReadHaloMergerTreeDescendant
    from RW_VRTools import BuildTemporalHeadTail
    from RW_VRTools import BuildTemporalHeadTailDescendant
    from RW_VRTools import ReadParticleDataFile
    from RW_VRTools import ReadSOParticleDataFile

##########################################################################################################################################################################
########################################################################### READ PARTICLE DATA ###########################################################################
##########################################################################################################################################################################

def read_sim_timesteps(run_directory,sim_type='SWIFT',snap_no=200,files_lz=4):
    ##### inputs
    # run directory: STRING for directory of run

    ##### returns
    # dictionary: lookback time, expansion factor, redshift for each snap (starting at snap = 0)

    if sim_type=='SWIFT':
        fields=['Lookback time [internal units]','Redshift','Scale-factor']
        prefix="snap_"

        snaps=[i for i in range(snap_no)]
        particle_data_file_directories=[run_directory+prefix+str(snap).zfill(files_lz)+".hdf5" for snap in snaps]

        sim_timesteps={'Lookback_time':[],'Redshift':[],'Scale_factor':[]}
        fields_out=list(sim_timesteps.keys())

        for snap in snaps:
            particle_file_temp=h5py.File(particle_data_file_directories[snap])
            time_unit=particle_file_temp['Units'].attrs['Unit time in cgs (U_t)']
            for ifield,field in enumerate(fields):
                if ifield==0:
                    sim_timesteps[fields_out[ifield]].extend(particle_file_temp['Cosmology'].attrs[field]*time_unit/(3600*24*365.25*10**9))
                else:
                    sim_timesteps[fields_out[ifield]].extend(particle_file_temp['Cosmology'].attrs[field])
            particle_file_temp.close()

    if sim_type=='GADGET':
        fields=['Redshift','Time']
        prefix="snapshot_"
        particle_file_temp=run_directory+prefix+str(0).zfill(files_lz)+".hdf5"
        particle_file_temp=h5py.File(particle_file_temp)

        H0=particle_file_temp['Header'].attrs['HubbleParam']*100
        Om0=particle_file_temp['Header'].attrs['Omega0']     
        cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

        snaps=[i for i in range(snap_no)]
        particle_data_file_directories=[run_directory+prefix+str(snap).zfill(files_lz)+".hdf5" for snap in snaps]

        sim_timesteps={'Redshift':np.zeros(snap_no),'Scale_factor':np.zeros(snap_no)}
        fields_out=list(sim_timesteps.keys())
        
        for snap in snaps:
            particle_file_temp=h5py.File(particle_data_file_directories[snap])
            for ifield,field in enumerate(fields):
                sim_timesteps[fields_out[ifield]][snap]=particle_file_temp['Header'].attrs[field]
            particle_file_temp.close()

        sim_timesteps['Lookback_time']=cosmo.lookback_time(sim_timesteps['Redshift']).astype(float)
        return sim_timesteps

def read_swift_particle_data(run_directory,snap_no=200,part_type=[0,1],data_fields=['Coordinates','Masses','ParticleIDs','Velocities'],verbose=1):
    
    ##### inputs
    # run_directory: STRING for directory of run
    # snap_no: integer number of snaps to record from snap=0
    # parttype: LIST of INTEGER desired particle type
    # datafields: LIST of swift particle fields from ['Coordinates','Masses','ParticleIDs','Velocities']

    ##### returns
    # dictionary (snaps) of dictionaries (corresponding to each parttype) of dictionaries (corresponding to fields) -- masses in solar units, velocities in km/s, positions in Mpc
    # e.g. particle_data[snap]['PartTypeX']['field'] where X is particle type integer

    snaps=[i for i in range(snap_no)]

    #initialise list
    particle_data=[[] for i in range(snap_no)]
    
    # load hdf5 files
    particle_data_file_directories=[run_directory+"snap_"+str(snap).zfill(4)+".hdf5" for snap in snaps]
    particle_data_files=[h5py.File(particle_data_file_directories[index]) for index in range(snap_no)]

    if verbose:
        print('Saving particle data')
    
    index=0
    # loop over snaps
    for snap in snaps:
        particle_data[snap]={}
        #loop over parttypes
        for part_type_temp in part_type:
            particle_data[snap]['PartType'+str(part_type_temp)]={}

            #loop over data fields
            for field in data_fields:
                if field=='Masses':
                    particle_data[snap]['PartType'+str(part_type_temp)][field]=np.array(particle_data_files[index]['PartType'+str(part_type_temp)][field])*10**10
                else:
                    particle_data[snap]['PartType'+str(part_type_temp)][field]=np.array(particle_data_files[index]['PartType'+str(part_type_temp)][field])
        index=index+1

    if verbose:
        print('Done saving particle data')

    #return dictionary of dictionary of dictionaries
    return particle_data

##########################################################################################################################################################################
############################################################################## CREATE HALO DATA ##########################################################################
##########################################################################################################################################################################

def read_vr_treefrog_data(vr_directory,vr_prefix,tf_directory,tf_name,files_type=2,files_nested=False,files_lz=4,snap_no=200,part_data_from_snap=120,extra_halo_fields=[],halo_TEMPORALHALOIDVAL=1000000,verbose=1):
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
    if files_nested==False:
        halo_data_all=[ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(files_lz),ibinary=files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True) for snap in sim_snaps]
    else:
        halo_data_temp=ReadPropertyFile(vr_directory+vr_prefix+str(200).zfill(files_lz)+"/"+vr_prefix+str(200).zfill(files_lz),ibinary=files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True)
        print(halo_data_temp)
        halo_data_all=[ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(files_lz)+"/"+vr_prefix+str(snap).zfill(files_lz),ibinary=files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True) for snap in sim_snaps]

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

    tf_treefile=tf_directory+tf_name
    print(tf_treefile)
    halo_tree=ReadHaloMergerTree(tf_treefile,ibinary=files_type,iverbose=1,imerit=True,inpart=False)
    BuildTemporalHeadTail(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=0,TEMPORALHALOIDVAL=halo_TEMPORALHALOIDVAL)

    if verbose==1:
        print('Finished assembling descendent tree using VR python tools')
    
    if verbose==1:
        print('Adding particle lists to halos using VR python tools')
    

    for snap in sim_snaps: #iterate through snaps to add particle data to halo_data_all structure

        n_halos=len(halo_data_all[snap]['ID'])

        if snap>part_data_from_snap:#if there are a good amount of halos and snap late enough
            if files_nested==False:
                part_data_temp=ReadParticleDataFile(vr_directory+vr_prefix+str(snap).zfill(files_lz),ibinary=files_type,iverbose=0,iparttypes=1)
            else:
                part_data_temp=ReadParticleDataFile(vr_directory+vr_prefix+str(snap).zfill(files_lz)+"/"+vr_prefix+str(snap).zfill(files_lz),ibinary=files_type,iverbose=0,iparttypes=1)

            for part_key in list(part_data_temp.keys()):
                halo_data_all[snap][part_key]=part_data_temp[part_key]

        else:#if we don't have particle data yet
            part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
            for part_key in list(part_data_temp.keys()):
                halo_data_all[snap][part_key]=part_data_temp[part_key] 

    if verbose==1:
        print('Finished adding particle lists to halos using VR python tools')

    if verbose==1:
        print('Appending FOF particle lists with substructure')

    for snap in sim_snaps:#iterate through snaps to add substructure particle lists to FOF halo particle

        halo_data_temp=halo_data_all[snap]
        field_halo_indices_temp=np.where(halo_data_temp['hostHaloID']==-1)[0]#find field/fof halos

        if len(field_halo_indices_temp)>0:#where there are field halos
            for field_halo_ID in halo_data_temp['ID'][field_halo_indices_temp]:#go through each field halo

                sub_halos_temp=(np.where(halo_data_temp['hostHaloID']==field_halo_ID)[0])#find its subhalos

                if len(sub_halos_temp)>0:#where there is substructure

                    field_halo_temp_index=np.where(halo_data_temp['ID']==field_halo_ID)[0][0]

                    field_halo_plist=halo_data_temp['Particle_IDs'][field_halo_temp_index]
                    field_halo_tlist=halo_data_temp['Particle_Types'][field_halo_temp_index]
                    sub_halos_plist=np.concatenate([halo_data_temp['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                    sub_halos_tlist=np.concatenate([halo_data_temp['Particle_Types'][isub] for isub in sub_halos_temp])#list all particles types substructure

                    halo_data_temp['Particle_IDs'][field_halo_temp_index]=np.concatenate([field_halo_plist,sub_halos_plist])#add particles to field halo particle list
                    halo_data_temp['Particle_Types'][field_halo_temp_index]=np.concatenate([field_halo_tlist,sub_halos_tlist])#add particles to field halo particle list
                    halo_data_temp['Npart'][field_halo_temp_index]=len(halo_data_temp['Particle_IDs'][field_halo_temp_index])#update Npart for each field halo

    if verbose==1:
        print('Finished appending FOF particle lists with substructure')

    return halo_data_all

##########################################################################################################################################################################
############################################################################## CREATE PARTICLE HISTORIES #################################################################
##########################################################################################################################################################################

def gen_part_history(halo_data,verbose=True):
    ##### inputs
    # halo_data (from above - needs particle lists)

    ##### returns
    # list (for each snap) of dictionaries (sub/field ids and types) of lists (for each bound particle) of:
    # (1) unique particle IDs of particles bound in field halos
    # (2) unique particle IDs of particles bound in subhalos

    sub_part_ids=[[] for i in range(len(halo_data))]
    field_part_ids=[[] for i in range(len(halo_data))]

    for snap in range(len(halo_data)):

        if verbose==True:
            print('Generating particle histories for snap = ',snap)

        if len(halo_data[snap]['Particle_IDs'])==0:#if no halos
            field_part_ids[snap]=[]
            sub_part_ids[snap]=[]

        else:
            field_halo_indices=np.where(halo_data[snap]['hostHaloID']==-1)[0]
            sub_halo_indices=np.where(halo_data[snap]['hostHaloID']>0)[0]

            #list of particles in field/subhalos for this snap
            field_part_ids_temp=[halo_data[snap]['Particle_IDs'][field_halo_index] for field_halo_index in field_halo_indices]
            sub_part_ids_temp=[halo_data[snap]['Particle_IDs'][sub_halo_index] for sub_halo_index in sub_halo_indices]

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

def gen_delta_npart(halo_data,sim_timesteps,type_order,mass_table,unique_particle_list=[],snaps=list(range(120,200,199)),depth=5,trim_hoes=True,verbose=True): 

    ##### inputs
    # halo_data (from above - needs particle lists)
    # snaps: list of snaps to consider
    # trim_hoes: get rid of particles which have been part of structure before when calculating accretion rate?

    ##### returns
    # halo_data with delta_m0 and delta_m1 keys 

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
    # if we done have particle lists, generate them now
    if unique_particle_list==[]:
        if verbose==True:
            print('Generating unique particle histories')
        unique_particle_list=gen_part_history(halo_data)

    # iterate through each snap
    for snap in snaps:
        if verbose==True:
            print('Generating accretion rates for snap = ',snap)
        
        #final snap particle data
        part_IDs_2=halo_data[snap]['Particle_IDs']
        part_Types_2=halo_data[snap]['Particle_Types']
        n_halo_2=len(halo_data[snap]['ID'])
        halo_tracked=np.zeros(n_halo_2)

        #initial snap particle data
        part_IDs_1=[[] for i in range(n_halo_2)]
        part_Types_1=[[] for i in range(n_halo_2)]

        #if initial particle lists are empty
        if halo_data[snap-depth]['Particle_IDs']==[] or snap-depth<0:
            if verbose==True:
                print('No particle lists found at initial snap = ',snap-depth)
            #record empty data
            for ihalo in range(n_halo_2):
                halo_tracked=False
                part_IDs_1[ihalo]=[]
                part_Types_1[ihalo]=[]
            continue

        #if we have initial particle list, find previous particle list
        else:
            for ihalo in range(n_halo_2):
                progen_index=find_progen_index(index_0=ihalo,snap=snap,depth=depth)
                if progen_index>-1:
                    part_IDs_1[ihalo]=halo_data[snap-depth]['Particle_IDs'][progen_index]
                    part_Types_1[ihalo]=halo_data[snap-depth]['Particle_Types'][progen_index]
                    halo_tracked[ihalo]=True
                else:
                    part_IDs_1[ihalo]=[]
                    part_Types_1[ihalo]=[]
                    halo_tracked[ihalo]=False

        #now have part_IDs_1 and part_IDs_2

        #now find accretion rate
        delta_n_tot=[np.nan for i in range(n_halo_2)]
        delta_m0=[np.nan for i in range(n_halo_2)]
        delta_m1=[np.nan for i in range(n_halo_2)]
        field_halo=[[] for i in range(n_halo_2)]

        substructure_partID_list_uptosnap=unique_particle_list[snap-1]["sub_ids"]
        structure_partID_list_uptosnap=unique_particle_list[snap-1]["field_ids"]

        for ihalo in range(n_halo_2):#for each z=0, find accretion rate
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
                m_gas=mass_table[1]*10**10 #MSol
                m_dm=mass_table[0]*10**10 #MSol

                if type_order==1:
                    delta_m0[ihalo]=np.sum(new_particle_types==0)*m_dm
                    delta_m1[ihalo]=np.sum(new_particle_types==1)*m_gas
                else:
                    delta_m0[ihalo]=np.sum(new_particle_types==1)*m_dm
                    delta_m1[ihalo]=np.sum(new_particle_types==0)*m_gas

        delta_t=abs(sim_timesteps['Lookback_time'][snap]-sim_timesteps['Lookback_time'][snap-depth])
        halo_data[snap]['delta_m0_dt']=delta_m0/delta_t
        halo_data[snap]['delta_m1_dt']=delta_m1/delta_t

    return halo_data
