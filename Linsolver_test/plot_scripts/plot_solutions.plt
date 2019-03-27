set term png
do for [t=0:26]{
  infile = sprintf('u_out_%i.dat',t)
  outfile = sprintf('u_out_%i.png',t)
  set output outfile
  plot infile matrix w image
}
set term wxt