#########################################################################################################################################################################
#########################################################################################################################################################################
####################################################### 01/04/2019 Ruby Wright - Python Data Processing Tools ###########################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

# packages preamble
import numpy as np

def flatten(list2d):

    """
    flatten : function
	----------
    Flatten a list of lists.
		
	Parameters
	----------
	list2d : list 
		List of lists to be flattened.

    Returns
	-------
	list2d_flattened : list
        Flattened version of input.
	
	"""

    ### input check
    
    if not type(list2d) == list:
        print("Please enter a list of lists to flatten.")                                                                              
        return []
        
    list2d_flattened=[]
    for list_temp in list2d:
        list2d_flattened.extend(list_temp)
    return list2d_flattened


def bin_xy(x,y,xy_mask=[],bins='eq',bin_range=[],n_per_bin=100,y_lop=16,y_hip=84,bin_min=5,verbose=False):

    """ 
    bin_xy : function
    ----------
    Find statistics of quantity y in bins of associated quantity x. 
		
	Parameters
	----------
	x : list or ndarray 
		independent variable to be binned
    
    y : list or ndarray 
		dependent variable (function of x) to calculate statistics for

    y_lop : float
        the lower percentile to calculate
    
    y_hip : float
        the upper percentile to calculate

    bin_min : int or float
        the lowest count of objects in a bin for which statistics will be calculated

    Returns
	-------
	bin_output : dict
        Dictionary of output statistics (all ndarray of length the number of bins):

        'bin_mid': midpoint of bins
        'bin_edges': edges of bins
        'Counts': histogram of x values in bins
        'Invalids': number of points deleted in each bin
        'Means': the means of binned y values
        'Medians': the medians of binned y values
        'Lo_P': the lower percentile of binned y values
        'Hi_P': the upper percentile of binned y values
	
	"""

    ### input processing and checking
    if not x.shape==y.shape:
        print('Please ensure x and y values have same length')
    if x.ndim>1 or y.ndim>1:
        print('Please enter 1-dimensional x and y values')
        return []
    if not (type(bins)==list or type(bins)==str or type(bins)==int or type(bins)==float):
        try:
            bins=np.array(bins)#convert to numpy array if manual
        except:
            print("Please enter bin edges of valid type")
    try:
        if xy_mask==[]:
            x=np.array(x) # convert x vals to numpy array
            y=np.array(y) # convert y vals to numpy array
        else:
            x=np.compress(xy_mask,x)
            y=np.compress(xy_mask,y)
    except:
        print("Please enter either a list or array for x and y values")
        return []

    try:
        bin_min=int(bin_min)
    except:
        print("Please enter valid bin minimum (should be 1-dimensional and of length 1)")
        return []

    ### create bins
    x_forbins=np.compress(np.logical_not(np.isnan(x)),x) # the x values to use in creating bins
    x_invalid=len(x)-len(x_forbins)

    if verbose:
        print(x_invalid,' x values had to be removed')

    if bins=='eq': # if default, calculate bin edges such that there are roughly the same count of data values in each bin (code from splotch)

        bin_no=np.floor(len(x_forbins)/n_per_bin).astype(int) # should be ~ 100 objects per bin
        b=bin_no
        if verbose:
            print("Generating bins using equal count method")
        if np.nanmin(x)==np.nanmax(x):
            bin_edges=np.linspace(np.nanmin(x)-0.5,np.nanmax(x)+0.5,num=b)
        else:
            from math import ceil,floor
            from numpy import array,concatenate,cumsum,ones,sort
            x_2=sort(x)
            L=len(x_2)
            w_l=floor(L/b)
            w_h=ceil(L/b)
            n_l=b*ceil(L/b)-L
            n_h=b-n_l
            n=concatenate([ones(ceil(n_l/2))*w_l,ones(n_h)*w_h,ones(floor(n_l/2))*w_l]).astype(int)
            if bin_range==[]:
                bin_edges=array([np.nanmin(x_2)]+[(x_2[i-1]+x_2[i])/2 for i in cumsum(n)[:-1]]+[np.nanmax(x_2)])
            else:
                bin_edges=array([bin_range[0]]+[(x_2[i-1]+x_2[i])/2 for i in cumsum(n)[:-1]]+[bin_range[1]])

            bin_mid=[]
            for ibin in range(bin_no):
                x_val_bin=np.compress(np.logical_and(x>bin_edges[ibin],x<bin_edges[ibin+1]),x)
                bin_mid.append(np.nanmedian(x_val_bin))

            if verbose:
                print("Bins generated, xmin = ",bin_edges[0],', xmax = ',bin_edges[-1],' and bin no = ',bin_no)

    elif type(bins)==int or type(bins)==float: # if given an integer number of bins, then create linear bins in x from 2nd to 98th percentile of finite values
        bin_no=int(bins)
        bin_edges=np.linspace(np.nanpercentile(x_forbins,2),np.nanpercentile(x_forbins,98),bin_no+1)
        bin_mid=np.array([bin_edges[ibin]+bin_edges[ibin+1] for ibin in range(bin_no)])*0.5
        if verbose:
            print('Number of bins given = ',bin_no)

    elif type(bins)==list or type(bins)==np.ndarray: # if given bin edges manually
        bin_no=len(bins)-1
        bin_mid=np.array([bins[i]+bins[i+1] for i in range(bin_no)])*0.5
        bin_edges=bins

        if verbose:
            print('Number of bins given = ',bin_no)


    #initialise outputs
    bin_init=np.zeros(bin_no)
    bin_output={'bin_mid':bin_mid,'bin_edges':bin_edges,'Counts':bin_init,'Invalids':bin_init,'Means':bin_init,'Medians':bin_init,'Lo_P':bin_init,'Hi_P':bin_init}

    means_temp=[]
    medians_temp=[]
    lops_temp=[]
    hips_temp=[]

    #notnan_mask=np.logical_and(np.logical_not(np.isnan(x)),np.logical_not(np.isnan(y)))#mask for all points which are notnan
    #finite_mask=np.logical_and(np.isfinite(x),np.isfinite(y))#mask for all points which are finite
    #valid_mask=np.logical_and(notnan_mask)#mask for all points which are both notnan and finite
    valid_mask=np.ones(len(x))

    for ibin,ibin_mid in enumerate(bin_mid):#loop through each bin
        
        bin_lo=bin_edges[ibin]#lower bin value
        bin_hi=bin_edges[ibin+1]#upper bin value
        
        bin_mask=np.logical_and(x>bin_lo,x<bin_hi)#mask for all points within x bin
        bin_count_gross=np.sum(bin_mask)
        bin_mask=np.logical_and(bin_mask,valid_mask)#mask for all valid points in x bin
        bin_count=np.sum(bin_mask)#count of selected objects

        x_subset=np.compress(bin_mask,np.array(x))
        y_subset=np.compress(bin_mask,np.array(y))

        mean_temp=np.nanmean(y_subset)#calculate mean
        median_temp=np.nanmedian(y_subset)#calculate median
        lop_temp=np.nanpercentile(y_subset,y_lop)#calculate lower percentile
        hip_temp=np.nanpercentile(y_subset,y_hip)#calculate upper percentile

        bin_output['Counts'][ibin]=bin_count#count of valid points in this bin
        bin_output['Invalids'][ibin]=bin_count_gross-bin_count#count of invalid points in this bin (probably something wrong with y vals)

        if bin_min==0 or bin_count>bin_min-1:
            means_temp.append(mean_temp)
            medians_temp.append(median_temp)
            lops_temp.append(lop_temp)
            hips_temp.append(hip_temp)        
        else:
            means_temp.append(np.nan)
            medians_temp.append(np.nan)
            lops_temp.append(np.nan)
            hips_temp.append(np.nan)
            if verbose:
                print("Insufficient count in bin at x = ",ibin_mid)

    bin_output['Means']=np.array(means_temp)
    bin_output['Medians']=np.array(medians_temp)
    bin_output['Lo_P']=np.array(lops_temp)
    bin_output['Hi_P']=np.array(hips_temp)

    return bin_output