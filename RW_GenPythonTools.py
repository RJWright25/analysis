#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################

#*** Packages ***

if True:
    import matplotlib.pyplot as plt
    import numpy as np

def flatten(array):
    result=[]
    for list_temp in array:
        result.extend(list_temp)
    return result

def bin_xy(x,y,bin_edges=[],y_lop=16,y_hip=84,bin_min=5):
    
    bin_no_default=10
    if bin_edges==[]:# if we're not given bins
        print('no bin edges given, generate default bin ranges')
        bin_no=bin_no_default
        bin_edges=np.linspace(np.nanpercentile(x,2),np.nanpercentile(x,98),bin_no+1)
        bin_mid=np.array([bin_edges[i]+bin_edges[i+1] for i in range(bin_no)])*0.5

    else:
        bin_no=len(bin_edges)-1
        bin_mid=np.array([bin_edges[i]+bin_edges[i+1] for i in range(bin_no)])*0.5
        print('number of bins given = ',bin_no)

    bin_init=np.zeros(bin_no)
    bin_output={'bin_mid':bin_mid,'bin_edges':bin_edges,'Counts':bin_init,'Means':bin_init,'Medians':bin_init,'Lo_P':bin_init,'Hi_P':bin_init}

    means_temp=[]
    medians_temp=[]
    lops_temp=[]
    hips_temp=[]

    for ibin,ibin_mid in enumerate(bin_mid):
        
        bin_lo=bin_edges[ibin]
        bin_hi=bin_edges[ibin+1]
        bin_mask=np.logical_and(x>bin_lo,x<bin_hi)
        bin_count=np.sum(bin_mask)

        x_subset=np.compress(bin_mask,np.array(x))
        y_subset=np.compress(bin_mask,np.array(y))

        mean_temp=np.nanmean(y_subset)
        median_temp=np.nanmedian(y_subset)
        lop_temp=np.nanpercentile(y_subset,y_lop)
        hip_temp=np.nanpercentile(y_subset,y_hip)

        bin_output['Counts'][ibin]=bin_count

        if bin_min==0 or bin_count>bin_min-1:
            count_true=True
        else:
            count_true=False

        if count_true:
            means_temp.append(mean_temp)
            medians_temp.append(median_temp)
            lops_temp.append(lop_temp)
            hips_temp.append(hip_temp)
        else:
            means_temp.append(np.nan)
            medians_temp.append(np.nan)
            lops_temp.append(np.nan)
            hips_temp.append(np.nan)

    bin_output['Means']=np.array(means_temp)
    bin_output['Medians']=np.array(medians_temp)
    bin_output['Lo_P']=np.array(lops_temp)
    bin_output['Hi_P']=np.array(hips_temp)

    for ibin in range(bin_count):
        for key in list(bin_output.keys()):
            if np.isinf(bin_output[key][ibin]):
                bin_output[key][ibin]=np.nan

    return bin_output
