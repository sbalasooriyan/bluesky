import os

Files = ['cfl.tex','los.tex','dep.tex','ipr.tex','time.tex','work.tex','density.tex',
         'dist.tex','timee.tex','worke.tex','densitye.tex','diste.tex']

#for File in Files:
#    os.system('pdflatex -quiet -output-directory=./plots/ ./plots/'+File)

for File in Files:
    os.system('copy "C:\\Users\\suthe\\University\\Courses Year 5\\AE5310 - Thesis Control and Operations\\bluesky\\additional_tools\\post_processor\\plots\\'+File+'" "C:\\Users\\suthe\\University\\Courses Year 5\\AE5310 - Thesis Control and Operations\\writing\\scientific_paper\\tikz\\plots"')