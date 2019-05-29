import glob
import numpy

flist=["/mnt/su3ctm/lbakels/CosmRun/9p/Hydro/nonrad/TreeFrogG/tree.snapshot_"+str(i).zfill(3)+".VELOCIraptor.tree" for i in range(201)]
print(flist)
numpy.savetxt("/mnt/su3ctm/rwright/hydro_accretion/LB_L32N512/treesnaplist.txt",flist)
