.PHONY: paper clean watch

paper: paper.pdf

paper.pdf: paper.tex references.bib
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf paper.toc

watch:
	fswatch -o paper.tex references.bib | xargs -n1 -I{} make paper
