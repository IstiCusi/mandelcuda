# Grundlegende Einstellungen
set terminal png size 800,600
set title 'Iteration Distribution'
set xlabel 'Iterations'
set ylabel 'Frequency'

# Bereich festlegen (falls gewünscht)
set xrange [0:10000]
# set yrange [MIN:MAX]

# Logarithmische Skala für den y-Achsen
set logscale y

# Breite der Bins für die Histogramme
binwidth=5
bin(x,width)=width*floor(x/width)+width/2.0

# Plotbefehle
set output 'distribution_normal.png'
set title 'Normal Distribution'
unset logscale y # Normale Skala für den ersten Plot
plot 'data.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes

set output 'distribution_logarithmic.png'
set title 'Logarithmic Distribution'
set logscale y # Logarithmische Skala für den zweiten Plot
plot 'data.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes

