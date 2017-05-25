
graficas_RC = Carga.pdf histograma_C.pdf histograma_R.pdf likelihood_C.pdf likeihood_R.pdf

Resultado_hw5.pdf : $(graficas_RC)
	pdflatex Resultados_hw5.tex

$(graficas_RC) : CircuitoRC.txt
	python circuitoRC.py

