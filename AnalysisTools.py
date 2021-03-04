#########################################################################################################################################################################
############################################################# Some miscellaneous python data analysis routines ##########################################################
#########################################################################################################################################################################

#### AUTHOR: Ruby Wright (Aug 2018)

# packages preamble
import numpy as np
from astropy.stats import bootstrap

def gen_bins(lo,hi,n,log=False):
    bin_output=dict()
    bin_output['edges']=np.linspace(lo,hi,n+1)
    bin_output['size']=bin_output['edges'][1]-bin_output['edges'][0]
    bin_output['mid']=(bin_output['edges']+bin_output['size']/2)[:n]
    bin_output['width']=np.array([bin_output['edges'][ibin+1]-bin_output['edges'][ibin] for ibin in range(n)])

    if log:
        bin_output['edges']=10**bin_output['edges']
        bin_output['mid']=10**bin_output['mid']
    return bin_output

def bin_xy(x,y,bins=None,bs=0,bin_min=5):

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
    x=np.array(x)
    y=np.array(y)

    if not x.shape==y.shape:
        print('Please ensure x and y values have same length')
    if x.ndim>1 or y.ndim>1:
        print('Please enter 1-dimensional x and y values')
        return []

    if type(bins)==list or type(bins)==np.ndarray: # if given bin edges manually
        bin_no=len(bins)-1
        bin_mid=np.array([bins[i]+bins[i+1] for i in range(bin_no)])*0.5
        bin_edges=bins
    else:
        bins=gen_bins(np.nanpercentile(x,1),np.nanpercentile(x,99),n=10)

    #initialise outputs
    bin_output={'bin_mid':bin_mid,
                'bin_edges':bin_edges,
                'bin_means':np.zeros(bin_no)+np.nan,
                'bin_medians':np.zeros(bin_no)+np.nan,
                'xfrac':np.nanmean(np.isfinite(x)),
                'yfrac':np.nanmean(np.isfinite(y)),
                'Counts':np.zeros(bin_no)+np.nan,
                'Invalids':np.zeros(bin_no)+np.nan,
                'Means':np.zeros(bin_no)+np.nan,
                'Medians':np.zeros(bin_no)+np.nan,
                'Lo_P':np.zeros(bin_no)+np.nan,
                'Hi_P':np.zeros(bin_no)+np.nan,
                'yerr-spread':np.zeros((2,bin_no))+np.nan,
                'Lo_P-2sigma':np.zeros(bin_no)+np.nan,
                'Hi_P-2sigma':np.zeros(bin_no)+np.nan,
                'yerr-spread':np.zeros((2,bin_no))+np.nan,
                'Sigma':np.zeros(bin_no)+np.nan,
                #the bootstrap outputs
                'bs_Lo_P_Median':np.zeros(bin_no)+np.nan, #95% CI
                'bs_Hi_P_Median':np.zeros(bin_no)+np.nan, #95% CI
                'yerr-median':np.zeros((2,bin_no))+np.nan,
                'bs_Sigma_Lo_P':np.zeros(bin_no)+np.nan, #95% CI
                'bs_Sigma_Hi_P':np.zeros(bin_no)+np.nan, #95% CI
                'yerr-sigma':np.zeros((2,bin_no))+np.nan,
                } 


    valid_mask=np.isfinite(y)
    for ibin in list(range(len(bin_mid))):#loop through each bin
        
        bin_lo=bin_edges[ibin]#lower bin value
        bin_hi=bin_edges[ibin+1]#upper bin value
        
        bin_mask=np.logical_and.reduce([x>bin_lo,x<bin_hi,valid_mask])
        bin_output['Counts'][ibin]=np.nansum(bin_mask)
        bin_output['Invalids'][ibin]=np.nansum(np.logical_and(x>bin_lo,x<bin_hi))-bin_output['Counts'][ibin]

        if bin_output['Counts'][ibin]>bin_min:
            x_subset=np.compress(bin_mask,np.array(x))
            y_subset=np.compress(bin_mask,np.array(y))

            bin_output['bin_means'][ibin]=np.nanmean(x_subset)
            bin_output['bin_medians'][ibin]=np.nanmedian(x_subset)
            bin_output['Means'][ibin]=np.nanmean(y_subset)
            bin_output['Medians'][ibin]=np.nanmedian(y_subset)
            bin_output['Sigma'][ibin]=np.nanstd(y_subset)
            bin_output['Lo_P'][ibin]=np.nanpercentile(y_subset,16)
            bin_output['Hi_P'][ibin]=np.nanpercentile(y_subset,84)
            bin_output['Lo_P-2sigma'][ibin]=np.nanpercentile(y_subset,2.275)
            bin_output['Hi_P-2sigma'][ibin]=np.nanpercentile(y_subset,97.725)
            bin_output['yerr-spread'][0,ibin]=bin_output['Medians'][ibin]-bin_output['Lo_P'][ibin]
            bin_output['yerr-spread'][1,ibin]=bin_output['Hi_P'][ibin]-bin_output['Medians'][ibin]
            
            if bs:                
                    median_sample=bootstrap(y_subset,bootnum=bs,samples=int(np.floor(len(y_subset)/2)),bootfunc=np.nanmedian)
                    bin_output['bs_Lo_P_Median'][ibin]=np.nanpercentile(median_sample,2.5)
                    bin_output['bs_Hi_P_Median'][ibin]=np.nanpercentile(median_sample,97.5)
                    bin_output['yerr-median'][0,ibin]=bin_output['Medians'][ibin]-bin_output['bs_Lo_P_Median'][ibin]
                    bin_output['yerr-median'][1,ibin]=bin_output['bs_Hi_P_Median'][ibin]-bin_output['Medians'][ibin]

                    sigma_sample=bootstrap(y_subset,bootnum=bs,samples=int(np.floor(len(y_subset)/2)),bootfunc=np.nanstd)
                    bin_output['bs_Lo_P_Sigma'][ibin]=np.nanpercentile(sigma_sample,2.5)
                    bin_output['bs_Hi_P_Sigma'][ibin]=np.nanpercentile(sigma_sample,97.5)
                    bin_output['yerr-sigma'][0,ibin]=bin_output['Sigma'][ibin]-bin_output['bs_Lo_P_Sigma'][ibin]
                    bin_output['yerr-sigma'][1,ibin]=bin_output['bs_Hi_P_Sigma'][ibin]-bin_output['Sigma'][ibin]

    return bin_output

def bin_2dimage(x,y,z,xedges,yedges,bin_min=5):
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
        ixbin_mask=np.logical_and(x>xlo,x<xhi)

        for iybin in range(nbins_y):
            ylo=yedges[iybin]
            yhi=yedges[iybin+1]
            iybin_mask=np.logical_and(y>ylo,y<yhi)
            ibin_mask=np.where(np.logical_and(ixbin_mask,iybin_mask))
            ibin_z=z[ibin_mask];z_count=len(ibin_z)
            
            output['Count'][iybin,ixbin]=z_count
            if z_count>bin_min:
                output['Means'][iybin,ixbin]=np.nanmean(ibin_z)
                output['Medians'][iybin,ixbin]=np.nanmedian(ibin_z)
                output['Lo_P'][iybin,ixbin]=np.nanpercentile(ibin_z,16)
                output['Hi_P'][iybin,ixbin]=np.nanpercentile(ibin_z,84)

    return output

def find_excess(x,y,xedges,ylog=False):
    x=np.array(x);y=np.array(y)
    median_relation=bin_xy(x=x,y=y,bins=xedges)['Medians']
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


# axes labels to reuse
axeslabels=dict()
