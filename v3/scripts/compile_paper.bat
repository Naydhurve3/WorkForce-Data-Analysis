@echo off
REM Compile the IEEE-format paper
REM Requires: pdflatex, bibtex

cd /d "%~dp0..\research_paper\04_paper_blueprint"

echo === Compiling IEEE paper (pass 1) ===
pdflatex -interaction=nonstopmode paper.tex 2>&1

echo === Running BibTeX ===
bibtex paper 2>&1

echo === Compiling IEEE paper (pass 2) ===
pdflatex -interaction=nonstopmode paper.tex 2>&1

echo === Compiling IEEE paper (pass 3) ===
pdflatex -interaction=nonstopmode paper.tex 2>&1

echo === Checking output ===
if exist paper.pdf (
    echo SUCCESS: paper.pdf created
) else (
    echo FAILURE: paper.pdf not found - check paper.log for errors
)

cd /d "%~dp0"
pause
