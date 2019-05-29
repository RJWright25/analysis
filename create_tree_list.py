import glob
import numpy

flist=glob.glob("/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/TreeFrogG/*.tree")
numpy.savetxt("/mnt/su3ctm/rwright/hydro_accretion/LB_L32N512/treesnaplist.txt",flist)
