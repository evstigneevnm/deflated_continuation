#set term eps
#set output 'BD.eps'
#set multiplot layout 2,1 title "Bifurcation diagram in different projections\n" font ",12"

set key below
#unset key
plot for [i=0:94] './'.i.'/debug_curve_all.dat' u 1:3 w d title ''.i

#set key below
#unset key
#plot for [i=0:94] './'.i.'/debug_curve.dat' u 1:3 w p ps 0.2 title ''.i

#unset multiplot
#set term qt
