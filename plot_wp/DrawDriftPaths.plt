set term png 
set output "printme.png"
set multiplot
set size ratio -1
set palette color positive
set pm3d map 
set pm3d interpolate 0,0 # makes prettier plots
set samples 50; set isosamples 50
set xtics font "Times-Bold,18"
set ytics font "Times-Bold,18"
set cbtics font "Times-Bold,18"
set rmargin 0 
set lmargin 0 
set tmargin 0 
set bmargin 0
set origin -0.03,0.0

set xrange[0:35]
set yrange[0:42]

set title ""
set xlabel "Radius (mm)" font "Times-Bold,20" offset 0,-1,0
set ylabel "Height (mm)" font "Times-Bold,20" offset -2,0,0
set cblabel "Weighting Potential" font "Times-Bold,20" offset 4,0,0
splot "test_wp.dat" using 1:2:3 notitle

unset multiplot
replot
set term x11