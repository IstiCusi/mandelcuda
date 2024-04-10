set terminal png size 800,600
set title 'Iteration Distribution'
set xlabel 'Iterations'
set ylabel 'Frequency'

set xrange [0:10000]
# set yrange [MIN:MAX]

set logscale y

binwidth=5
bin(x,width)=width*floor(x/width)+width/2.0

set output 'distribution_normal.png'
set title 'Normal Distribution'
unset logscale y
plot 'distribution.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes

set output 'distribution_logarithmic.png'
set title 'Logarithmic Distribution'
set logscale y 
plot 'distribution.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes

