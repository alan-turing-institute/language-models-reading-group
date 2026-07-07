.PHONY: clean all

all: maths.pdf

clean:
	latexmk -c maths.tex

maths.pdf : maths.tex
	latexmk -pdflua $<
