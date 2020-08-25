########################################################################################################################################################################
#########################################################################################################################################################################
####################################################### 01/04/2019 Ruby Wright - Python Data Processing Tools ###########################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

# packages preamble
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from astropy.stats import bootstrap
from matplotlib.colors import ListedColormap

def gen_bins(lo,hi,n,log=False,symlog=False):
    bin_output=dict()
    bin_output['edges']=np.linspace(lo,hi,n+1)
    bin_output['size']=bin_output['edges'][1]-bin_output['edges'][0]
    bin_output['mid']=(bin_output['edges']+bin_output['size']/2)[:n]
    bin_output['width']=np.array([bin_output['edges'][ibin+1]-bin_output['edges'][ibin] for ibin in range(n)])

    if log:
        bin_output['edges']=10**bin_output['edges']
        bin_output['mid']=10**bin_output['mid']
        if symlog:
            bin_output['edges']=np.array(flatten([-bin_output['edges'][::-1],bin_output['edges']]))
            bin_output['mid']=[bin_output['edges'][i+1]*0.5+bin_output['edges'][i]*0.5 for i in range(len(bin_output['edges'])-1)]
            
    return bin_output

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

def bin_xy(x,y,xy_mask=[],bins=[],bin_range=[],y_lop=16,y_hip=84,bs=0,bin_min=5,verbose=False):

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


    if type(bins)==int or type(bins)==float: # if given an integer number of bins, then create linear bins in x from 2nd to 98th percentile of finite values
        bin_no=int(bins)
        if bin_range==[]:
            bin_edges=np.linspace(np.nanpercentile(x_forbins,2),np.nanpercentile(x_forbins,98),bin_no+1)
        else:
            bin_edges=np.linspace(bin_range[0],bin_range[1],bin_no+1)
            
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
    bin_output={'bin_mid':bin_mid,'bin_edges':bin_edges,'bin_means':np.zeros(bin_no),'bin_medians':np.zeros(bin_no),'Counts':np.zeros(bin_no),'Invalids':np.zeros(bin_no),'Means':np.zeros(bin_no),'Medians':np.zeros(bin_no),'Lo_P':np.zeros(bin_no),'Hi_P':np.zeros(bin_no),'bs_Lo_P_Median':np.zeros(bin_no)+np.nan,'bs_Hi_P_Median':np.zeros(bin_no)+np.nan}

    bin_counts_temp=[]
    bin_invalids_temp=[]
    xmeans_temp=[]
    xmedians_temp=[]
    means_temp=[]
    medians_temp=[]
    lops_temp=[]
    hips_temp=[]
    yerrs_temp=[]

    finite_mask=np.logical_and(np.isfinite(x),np.isfinite(y))#mask for all points which are notnan
    valid_mask=finite_mask#mask for all points which are both notnan and finite
    
    for ibin,ibin_mid in enumerate(bin_mid):#loop through each bin
        
        bin_lo=bin_edges[ibin]#lower bin value
        bin_hi=bin_edges[ibin+1]#upper bin value
        
        bin_mask=np.logical_and(x>bin_lo,x<bin_hi).astype(int)#mask for all points within x bin
        bin_count_gross=np.nansum(bin_mask)
        bin_mask=np.logical_and(bin_mask,valid_mask)
        bin_count=np.nansum(bin_mask)#count of selected objects (where y is also notnan)

        x_subset=np.compress(bin_mask,np.array(x))
        y_subset=np.compress(bin_mask,np.array(y))


        if bs:
            if ibin==0:
                print('Doing bootstrap...')
            if bin_count>5:                    
                sample_results=bootstrap(y_subset,bootnum=bs,samples=int(np.floor(len(y_subset)/2)),bootfunc=np.nanmedian)
                bs_lo_p=np.nanpercentile(sample_results,16)
                bs_hi_p=np.nanpercentile(sample_results,84)
                bin_output['bs_Lo_P_Median'][ibin]=bs_lo_p
                bin_output['bs_Hi_P_Median'][ibin]=bs_hi_p
            else:
                bin_output['bs_Lo_P_Median'][ibin]=np.nan
                bin_output['bs_Hi_P_Median'][ibin]=np.nan

        xmean_temp=np.nanmean(x_subset)
        xmedian_temp=np.nanmedian(x_subset)

        mean_temp=np.nanmean(y_subset)#calculate mean
        median_temp=np.nanmedian(y_subset)#calculate median
        lop_temp=np.nanpercentile(y_subset,y_lop)#calculate lower percentile
        hip_temp=np.nanpercentile(y_subset,y_hip)#calculate upper percentile
        yerr_temp=[median_temp-lop_temp,hip_temp-median_temp]#yerr for errbar

        bin_counts_temp.append(bin_count)
        bin_invalids_temp.append(bin_count_gross-bin_count)

        if bin_min==0 or bin_count>bin_min-1:
            xmeans_temp.append(xmean_temp)
            xmedians_temp.append(xmedian_temp)
            means_temp.append(mean_temp)
            medians_temp.append(median_temp)
            lops_temp.append(lop_temp)
            hips_temp.append(hip_temp)        
            yerrs_temp.append(yerr_temp)        
        else:
            xmeans_temp.append(np.nan)
            xmedians_temp.append(np.nan)
            means_temp.append(np.nan)
            medians_temp.append(np.nan)
            lops_temp.append(np.nan)
            hips_temp.append(np.nan)
            yerrs_temp.append([np.nan,np.nan])
            if verbose:
                print("Insufficient count in bin at x = ",ibin_mid)

    bin_output['Counts']=np.array(bin_counts_temp)
    bin_output['Invalids']=np.array(bin_invalids_temp)
    bin_output['bin_means']=np.array(xmeans_temp)
    bin_output['bin_medians']=np.array(xmedians_temp)
    bin_output['Means']=np.array(means_temp)
    bin_output['Medians']=np.array(medians_temp)
    bin_output['Lo_P']=np.array(lops_temp)
    bin_output['Hi_P']=np.array(hips_temp)
    bin_output['yerr']=np.transpose(np.array(yerrs_temp))

    return bin_output

def bin_2dimage(x,y,z,xedges,yedges):
    nbins_x=len(xedges)-1
    nbins_y=len(yedges)-1
    nbins=nbins_x*nbins_y
    output={'Means':np.zeros((nbins_y,nbins_x)),'Medians':np.zeros((nbins_y,nbins_x)),'Lo_P':np.zeros((nbins_y,nbins_x)),'Hi_P':np.zeros((nbins_y,nbins_x)),'Count':np.zeros((nbins_y,nbins_x))}
    for ixbin in range(nbins_x):
        xlo=xedges[ixbin]
        xhi=xedges[ixbin+1]
        xmid=(xhi+xlo)/2
        ixbin_mask=np.logical_and(x>xlo,x<xhi)

        for iybin in range(nbins_y):
            ylo=yedges[iybin]
            yhi=yedges[iybin+1]
            ymid=(yhi+ylo)/2
            iybin_mask=np.logical_and(y>ylo,y<yhi)
            ibin_mask=np.where(np.logical_and(ixbin_mask,iybin_mask))
            ibin_z=z[ibin_mask]

            z_count=len(ibin_z);output['Count'][iybin,ixbin]=z_count
            z_mean=np.nanmean(ibin_z);output['Means'][iybin,ixbin]=z_mean
            z_median=np.nanmedian(ibin_z);output['Medians'][iybin,ixbin]=z_median
            z_lop=np.nanpercentile(ibin_z,16);output['Lo_P'][iybin,ixbin]=z_lop
            z_hip=np.nanpercentile(ibin_z,84);output['Hi_P'][iybin,ixbin]=z_hip

    return output

def get_methods(object, spacing=20): 
  methodList = [] 
  for method_name in dir(object): 
    try: 
        if callable(getattr(object, method_name)): 
            methodList.append(str(method_name)) 
    except: 
        methodList.append(str(method_name)) 
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s) 
  for method in methodList: 
    try: 
        print(str(method.ljust(spacing)) + ' ' + 
              processFunc(str(getattr(object, method).__doc__)[0:90])) 
    except: 
        print(method.ljust(spacing) + ' ' + ' getattr() failed')

def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    low = c(low)
    high = c(high)
    return make_colormap([low, high])
    
def diverge_map_grey(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    low = c(low)
    high = c(high)
    return make_colormap([low,c('dimgrey'), high])

def diverge_map_list(clist):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    colorlist=[c(col) for col in clist]
    return make_colormap(colorlist)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

RW_axlabs={
'M200':r'$M_{\rm 200,\ crit}/M_{\odot}$',
'accrate_bar':r'$\frac{\Delta M_{\rm baryon,\ FOF}\ ({\rm inflow})}{\Delta t}$',
'acceff_bar':r'$\frac{\Delta M_{\rm baryon,\ FOF}/{\Delta t}\ ({\rm inflow})}{f_{\rm b}\ M_{\rm 200}}\ [{\rm Gyr}^{-1}]$',
'deltam_norm':r'$\frac{{\Delta M_{\rm baryon,\ FOF}\ ({\rm inflow})-\Delta M_{\rm baryon,\ FOF}\ ({\rm outflow})}/{\Delta t}}{M_{\rm 200,\ crit}}$'
}

def make_cmap_totrans(c):
    c = list(mcolors.ColorConverter().to_rgba(c))
    clist=[c]*256
    for iic,ic in enumerate(clist):
        ic=np.array(ic)
        ic[3]=iic/len(clist)
        clist[iic]=tuple(ic)
    newcmp=ListedColormap(clist)
    return newcmp
def make_cmap_toblk(c):
    c = list(mcolors.ColorConverter().to_rgba(c))
    clist=[c]*256
    for iic,ic in enumerate(clist):
        ic=np.array(ic)
        ic[:3]=iic/len(clist)*ic[:3]
        clist[iic]=tuple(ic)
    newcmp=ListedColormap(clist)
    return newcmp


#find excess function
def find_excess(x,y,x_edges):
    mean_ys=bin_xy(x=x,y=y,bins=x_edges)['Medians']
    x_bin=np.zeros(len(y)).astype(int)-1
    for iix,ix in enumerate(x):
        whichbin=np.where(np.logical_and(ix>x_edges[:-1],ix<x_edges[1:]))
        if len(whichbin[0])==1:
            x_bin[iix]=whichbin[0][0]
    mean_y_forx=np.array([mean_ys[ix_bin] for ix_bin in x_bin])
    y_normalised=y/(mean_y_forx+1e-10)
    return y_normalised