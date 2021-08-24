
#                  _         _    
#   __ _ _ _  __ _| |_  _ __(_)___
#  / _` | ' \/ _` | | || (_-< (_-<
#  \__,_|_||_\__,_|_|\_, /__/_/__/
#                    |__/         

# Ruby Wright, 2018
# Code for frequently used methods & data analysis.

# Imports
import numpy as np
import pickle
from astropy.stats import bootstrap

# Generate bins for histograms, binned_statistic etc
def gen_bins(lo,hi,n=10,log=False):

    """ 
        gen_bins : function
        ----------
        Generate bins for statistics with specified interval. 
            
        Parameters
        ----------
        lo : float 
            lower bound of parameter range
        hi : float 
            upper bound of parameter range
        n : int
            number of bin midpoints
        log : bool
            assume lo and hi are log10 base values, convert to regular space

        Returns
        -------
        bin_output : dict
            Dictionary of bin details:
            'mid': midpoints of bins (length n)
            'edges': edges of bins (length n+1)
            'width': width of each bin (length n)

    """

    bin_output=dict()
    bin_output['edges']=np.linspace(lo,hi,n+1)
    bin_output['mid']=(bin_output['edges'][:-1]+bin_output['edges'][1:])/2
    bin_output['width']=np.array([bin_output['edges'][ibin+1]-bin_output['edges'][ibin] for ibin in range(n)])

    if log:
        bin_output['edges']=10**bin_output['edges']
        bin_output['mid']=10**bin_output['mid']

    return bin_output

# Calculate binned statistic for 2D data with specified bins
def bin_2d(x,y,bins=None,pciles=None,bin_min=5,bs=0):

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
    bins : dict, list or ndarray
        the bins to use for calculating stats
    pciles : list of floats
        the percentile values to calculate 
    bin_min : int or float
        the lowest count of objects in a bin for which statistics will be calculated
    bs : int 
        the number of bootstrap resamples to be conducted (zero if none)
    
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
        'nP': the nth percentile of binned y values

        if bootstrap is requested:
            'bs_*_lo_1sigma': the 16th percentile of bootstrapped * stat y values
            'bs_*_hi_1sigma': the 84th percentile of bootstrapped * stat y values
            'bs_*_lo_2sigma': the 2.5th percentile of bootstrapped * stat y values
            'bs_*_hi_2sigma': the 97.5th percentile of bootstrapped * stat y values

	"""

    ### input processing and checking
    x=np.array(x)
    y=np.array(y)

    outputs=['bin_Means','bin_Medians','Means','Medians','Sigma','Counts','Invalids']

    if not x.shape==y.shape:
        print('Please ensure x and y values have same length')
    if x.ndim>1 or y.ndim>1:
        print('Please enter 1-dimensional x and y values')
        return None

    if type(bins)==dict:
        bin_mid=bins['mid']
        bin_edges=bins['edges']
    if type(bins)==list or type(bins)==np.ndarray: # if given bin edges manually
        bin_mid=np.array([bins[i]+bins[i+1] for i in range(len(bins)-1)])*0.5
        bin_edges=bins
    else:
        bin_edges=gen_bins(np.nanpercentile(x,1),np.nanpercentile(x,99),n=11)
        bin_mid=np.array([bin_edges[i]+bin_edges[i+1] for i in range(len(bin_edges)-1)])*0.5

    bin_no=len(bin_mid)

    if not pciles:
        pciles=[1,2.5,5,10,16,20,30,40,50,60,70,80,84,90,95,97.5,99]
    pcile_names=[]

    bs_pcile_idxs=[8]

    for pcile in pciles:
        pcile_name=str(int(pcile))+'P'
        pcile_names.append(pcile_name)
        outputs.append(pcile_name)

    
    #initialise outputs
    bin_output=dict()
    for field in outputs:
        bin_output[field]=np.zeros(bin_no)+np.nan

    bin_output['bin_mid']=bin_mid
    bin_output['bin_edges']=bin_edges

    if bs:
        bs_outputs=['bs_Means_lo_1sigma','bs_Means_hi_1sigma','bs_Means_lo_2sigma','bs_Means_hi_2sigma','bs_Sigma_lo_1sigma','bs_Sigma_hi_1sigma','bs_Sigma_lo_2sigma','bs_Sigma_hi_2sigma']

        for bs_output in bs_outputs:
            bin_output[bs_output]=np.zeros(bin_no)+np.nan

        for ipcile in bs_pcile_idxs:
            pcile=pciles[ipcile]
            pcile_name=pcile_names[ipcile]
            bin_output['bs_'+pcile_name+'_lo_1sigma']=np.zeros(bin_no)+np.nan
            bin_output['bs_'+pcile_name+'_hi_1sigma']=np.zeros(bin_no)+np.nan
            bin_output['bs_'+pcile_name+'_lo_2sigma']=np.zeros(bin_no)+np.nan
            bin_output['bs_'+pcile_name+'_hi_2sigma']=np.zeros(bin_no)+np.nan

    y_good=np.isfinite(y)
    y_bad=np.logical_not(y_good)

    for ibin,(bin_lo,bin_hi) in enumerate(zip(bin_edges[:-1],bin_edges[1:])):#loop through each bin
        
        bin_lo=bin_edges[ibin]#lower bin value
        bin_hi=bin_edges[ibin+1]#upper bin value
        
        bin_mask=np.logical_and.reduce([x>bin_lo,x<bin_hi])
        bin_output['Counts'][ibin]=np.nansum(bin_mask)
        bin_output['Invalids'][ibin]=np.nansum(np.logical_and(bin_mask,y_bad))

        if bin_output['Counts'][ibin]>bin_min:
            bin_valid=np.logical_and(y_good,bin_mask)
            counts=np.sum(bin_valid)

            x_subset=np.array(x[bin_valid])
            y_subset=np.array(y[bin_valid])

            bin_output['bin_Means'][ibin]=np.mean(x_subset)
            bin_output['bin_Medians'][ibin]=np.median(x_subset)

            bin_output['Means'][ibin]=np.mean(y_subset)
            bin_output['Medians'][ibin]=np.median(y_subset)
            bin_output['Sigma'][ibin]=np.std(y_subset)

            for pcile,pcile_name in zip(pciles,pcile_names):
                bin_output[pcile_name][ibin]=np.nanpercentile(y_subset,pcile)

            if bs:
                num_samples=int(np.floor(counts/2))
                for ipcile in bs_pcile_idxs:
                    pcile=pciles[ipcile]
                    pcile_name=pcile_names[ipcile]

                    def pcile_func(y_subset):
                        return np.nanpercentile(y_subset,pcile)

                    pcile_sample=bootstrap(y_subset,bootnum=bs,samples=num_samples,bootfunc=pcile_func)
                    bin_output['bs_'+pcile_name+'_lo_1sigma'][ibin]=np.nanpercentile(pcile_sample,16)
                    bin_output['bs_'+pcile_name+'_hi_1sigma'][ibin]=np.nanpercentile(pcile_sample,84)
                    bin_output['bs_'+pcile_name+'_lo_2sigma'][ibin]=np.nanpercentile(pcile_sample,2.5)
                    bin_output['bs_'+pcile_name+'_hi_2sigma'][ibin]=np.nanpercentile(pcile_sample,97.5)

                for stat,func in zip(['Means','Sigma'],[np.nanmean,np.nanstd]):
                    stat_sample=bootstrap(y_subset,bootnum=bs,samples=num_samples,bootfunc=func)
                    bin_output['bs_'+stat+'_lo_1sigma'][ibin]=np.nanpercentile(stat_sample,16)
                    bin_output['bs_'+stat+'_hi_1sigma'][ibin]=np.nanpercentile(stat_sample,84)
                    bin_output['bs_'+stat+'_lo_2sigma'][ibin]=np.nanpercentile(stat_sample,2.5)
                    bin_output['bs_'+stat+'_hi_2sigma'][ibin]=np.nanpercentile(stat_sample,97.5)

    return bin_output

# Calculate binned statistic for 3D data with specified bins
def bin_3d(x,y,z,xedges,yedges,bin_min=5):
    
    nbins_x=len(xedges)-1
    nbins_y=len(yedges)-1
    nbins=nbins_x*nbins_y

    output={'Means':np.zeros((nbins_y,nbins_x))+np.nan,
            'Medians':np.zeros((nbins_y,nbins_x))+np.nan,
            'Lo_P':np.zeros((nbins_y,nbins_x))+np.nan,
            'Hi_P':np.zeros((nbins_y,nbins_x))+np.nan,
            'Count':np.zeros((nbins_y,nbins_x))+np.nan}

    for ixbin in range(nbins_x):
        xlo=xedges[ixbin]
        xhi=xedges[ixbin+1]
        ixbin_mask=np.logical_and(x>=xlo,x<xhi)

        for iybin in range(nbins_y):
            ylo=yedges[iybin]
            yhi=yedges[iybin+1]
            iybin_mask=np.logical_and(y>=ylo,y<yhi)
            ibin_mask=np.where(np.logical_and(ixbin_mask,iybin_mask))
            ibin_z=z[ibin_mask];z_count=len(ibin_z)
            
            output['Count'][iybin,ixbin]=z_count
            if z_count>bin_min:
                output['Means'][iybin,ixbin]=np.nanmean(ibin_z)
                output['Medians'][iybin,ixbin]=np.nanmedian(ibin_z)
                output['Lo_P'][iybin,ixbin]=np.nanpercentile(ibin_z,16)
                output['Hi_P'][iybin,ixbin]=np.nanpercentile(ibin_z,84)

    return output

# Calculate excess value of parameter relative to control set in 2d 
def find_excess(x,y,xedges,xmatch=False,ymatch=False,ylog=False):
    x=np.array(x);y=np.array(y)
    if not np.sum(np.logical_and(xmatch,ymatch)):
        xmatch=x;ymatch=y
    median_relation=bin_2d(x=xmatch,y=ymatch,bins=xedges,bin_min=20)['Medians']
    
    bin_allocation=np.zeros(len(y)).astype(int)-1
    for ix,xval in enumerate(x):
        ix_binallocation=np.where(np.logical_and(xval>xedges[:-1],xval<xedges[1:]))
        if len(ix_binallocation[0])==1:
            bin_allocation[ix]=ix_binallocation[0][0]
    matched_medians=np.array([median_relation[ix_bin] for ix_bin in bin_allocation])
    
    if not ylog:
        y_normalised=y/matched_medians
    else:
        y_normalised=y-matched_medians

    return y_normalised

# Calculate excess value of parameter relative to control set in 3d 
def find_excess_2d(x,y,z,xedges,yedges,xmatch=None,ymatch=None,zmatch=None,zlog=False,use='Median'):

    if not np.sum(zmatch):
        xmatch=x
        ymatch=y
        zmatch=z

    x=np.array(x);y=np.array(y);z=np.array(z)
    median_relation=bin_2d(xmatch,ymatch,zmatch,xedges,yedges)[use]

    bin_allocation_x=np.zeros(len(z)).astype(int)-1
    bin_allocation_y=np.zeros(len(z)).astype(int)-1
    
    for ival,(xval,yval) in enumerate(zip(x,y)):
        ival_x_binallocation=np.where(np.logical_and(xval>xedges[:-1],xval<xedges[1:]))
        ival_y_binallocation=np.where(np.logical_and(yval>yedges[:-1],yval<yedges[1:]))
        if len(ival_x_binallocation[0])==1:
            bin_allocation_x[ival]=ival_x_binallocation[0][0]
        if len(ival_y_binallocation[0])==1:
            bin_allocation_y[ival]=ival_y_binallocation[0][0]

    matched_medians=np.array([median_relation[iy_bin][ix_bin] for ix_bin,iy_bin in zip(bin_allocation_x,bin_allocation_y)])
    
    if not zlog:
        z_normalised=z/matched_medians
    else:
        z_normalised=z-matched_medians

    return z_normalised

# Open a pickled binary file
def open_pickle(path):
    """

    open_pickle : function
	----------------------

    Open a (binary) pickle file at the specified path, close file, return result.

	Parameters
	----------
    path : str
        Path to the desired pickle file. 


    Returns
	----------
    output : data structure of desired pickled object

    """

    with open(path,'rb') as picklefile:
        pickledata=pickle.load(picklefile)
        picklefile.close()

    return pickledata

# Dump object to a binary file
def dump_pickle(data,path):
    """

    dump_pickle : function
	----------------------

    Dump data to a (binary) pickle file at the specified path, close file.

	Parameters
	----------
    data : any type
        The object to pickle. 

    path : str
        Path to the desired pickle file. 


    Returns
	----------
    None

    Creates a file containing the pickled object at path. 

    """

    with open(path,'wb') as picklefile:
        pickle.dump(data,picklefile,protocol=4)
        picklefile.close()
    return data