'''
output.py

This script imports the processed data and outputs it.
'''

import numpy as np
import os
import pickle
import pandas
import matplotlib.pyplot as plt
from dataclasses import ProcessedData, ACData, CFLLOS
from scipy.stats import norm

def replaceColNames(colnames):
    for i in range(len(colnames)):
        colnames[i] = colnames[i].replace('-OFF','')
        colnames[i] = colnames[i].replace('SSD-FF','SSD')
    return colnames

def stats(entry, confidence):
    z = norm.ppf(1-(1-float(confidence)/100)/2)
    mean = np.mean(entry)
    median = np.median(entry)
    stdev = np.std(entry)
    n = entry.shape[0]
    
    # Box plot data
    upper_quartile = np.percentile(entry, 75)
    lower_quartile = np.percentile(entry, 25)    
    iqr = upper_quartile - lower_quartile
    upper_whisker = entry[entry<=upper_quartile+1.5*iqr].max()
    lower_whisker = entry[entry>=lower_quartile-1.5*iqr].min()
    
    # Confidence data
    upper_lim = mean + z * stdev / np.sqrt(float(n))
    lower_lim = mean - z * stdev / np.sqrt(float(n))
    
    stats = np.array([mean,median,stdev,n,upper_quartile,lower_quartile,
                      upper_whisker,lower_whisker,upper_lim,lower_lim])

    return stats

def confplot(colnames,data,inst_labels,ylabel,ylim,outDir,outName):
    
    markopt = {0: "mark=*", 1: "mark=square*", 2: "mark=triangle*", 3: "mark=diamond*"}
    if len(inst_labels) == 3:
        colopt = {2: "bluevec", 1: "greenvec", 0: "redvec"}
    elif len(inst_labels) == 2:
        colopt = {1: "bluevec", 0: "greenvec"}
    # Output
    lines = []
    
    # Preamble
    lines.append("\\documentclass{standalone}\n")
    lines.append("\\usepackage{xcolor}\n")
    lines.append("\\usepackage{tikz}\n")
    lines.append("\\usetikzlibrary{plotmarks}\n")
    lines.append("\\usepackage{pgfplots}\n")
    lines.append("\\pgfplotsset{compat=1.12}\usepgfplotslibrary{statistics}\n")
    lines.append("\\renewcommand{\\sfdefault}{phv}\n")
    lines.append("\\renewcommand{\\rmdefault}{ptm}\n")
    lines.append("\\renewcommand{\\ttdefault}{pcr}\n")
    lines.append("\definecolor{redvec}{RGB}{192,0,0}\n")
    lines.append("\definecolor{greenvec}{RGB}{0,192,0}\n")
    lines.append("\definecolor{bluevec}{RGB}{0,0,192}\n")
    lines.append("\definecolor{orangevec}{RGB}{192,96,0}\n")
    # Start document and tikz
    lines.append("\\begin{document}\n")
    lines.append("    \\begin{tikzpicture}\n")
    # Border (clip)
    lines.append("        \\path[clip] (-1.15,-0.33) rectangle (7.67,4.63);\n")
    lines.append("        \\begin{axis}[height=165,\n")
    lines.append("                     width=262.5,\n")
    lines.append("                     ymajorgrids,\n")
    lines.append("                     xmin=0.25,\n")
    lines.append("                     xmax=%s,\n" % str(float(len(colnames))+0.75))
    if ylim[0] != None:
        lines.append("                     ymin=%s,\n" % str(ylim[0]))
    if ylim[1] != None:
        lines.append("                     ymax=%s,\n" % str(ylim[1]))
    lines.append("                     label style={font=\\footnotesize},\n")
    lines.append("                     tick label style={font=\\footnotesize,major tick length=0,/pgf/number format/fixed},\n")
    lines.append("                     ylabel={%s},\n" % ylabel)
    lines.append("                     legend style={at={(0.5,1.0)},anchor=south,font=\\footnotesize,draw=none,fill=none,/tikz/every even column/.append style={column sep=20}},\n")
    lines.append("                     legend columns=%s,\n" % str(len(inst_labels)))
    lines.append("                     xtick={")
    lines.append("                     xticklabels={")
    lines.append("                     minor xtick={0.5,")
    for i in range(len(colnames)):
        lines[-3] += str(i+1)
        lines[-1] += str(float(i+1.5))
        if colnames[i] == "NO CR":
            lines[-2] += "\\,\\,"
        lines[-2] += colnames[i]
        if i+1 != len(colnames):
            lines[-3] += ","
            lines[-2] += ","
            lines[-1] += ","
    lines[-3] += "},\n"
    lines[-2] += "},\n"
    lines[-1] += "},\n"
    lines.append("                     minor grid style={dotted,lightgray},\n")
    lines.append("                     xminorgrids]\n")

    for i in range(len(colnames)):
        for j in range(len(data[i])):
            lines.append("            \\addplot [only marks, %s, fill=%s, draw=%s, error bars/.cd, y dir=both, y explicit, error mark options={%s, rotate=90, mark size=2}] coordinates {\n" % (markopt[j],colopt[j],colopt[j],colopt[j]))
            lines.append("                (%s,%s) += (0,%s) -= (0,%s)};\n" % (str(i+0.5+(j+1)*1./(len(data[i])+1)), str(data[i][j][0]), str(data[i][j][8]-data[i][j][0]), str(data[i][j][0]-data[i][j][9]))) 

    lines.append("            \\legend{")
    for i in range(len(inst_labels)):
        lines[-1] += inst_labels[i]
        if i+1 != len(inst_labels):
            lines[-1] += ","
        else:
            lines[-1] += "}\n"
    lines.append("        \\end{axis}\n")
    lines.append("    \\end{tikzpicture}\n")
    lines.append("\\end{document}\n")
	

    # Write output
    scnFileName = os.path.join(outDir, "bar_"+outName) 
    f           = open(scnFileName,"w")
    f.writelines(lines)
    f.close()

def boxplot(colnames,data,inst_labels,ylabel,ylim,outDir,outName,fill=1):
    if len(inst_labels) == 3:
        colopt = {2: "bluevec", 1: "greenvec", 0: "redvec"}
        if fill:
            filler = [", fill=redvecl",", fill=greenvecl",", fill=bluevecl"]
        else:
            filler = ["","",""]
    elif len(inst_labels) == 2:
        colopt = {1: "bluevec", 0: "greenvec"}
        if fill:
            filler = [", fill=greenvecl",", fill=bluevecl"]
        else:
            filler = ["",""]
    # Output
    lines = []
    
    # Preamble
    lines.append("\\documentclass{standalone}\n")
    lines.append("\\usepackage{xcolor}\n")
    lines.append("\\usepackage{tikz}\n")
    lines.append("\\usetikzlibrary{plotmarks}\n")
    lines.append("\\usepackage{pgfplots}\n")
    lines.append("\\pgfplotsset{compat=1.12}\usepgfplotslibrary{statistics}\n")
    lines.append("\\renewcommand{\\sfdefault}{phv}\n")
    lines.append("\\renewcommand{\\rmdefault}{ptm}\n")
    lines.append("\\renewcommand{\\ttdefault}{pcr}\n")
    lines.append("\definecolor{redvec}{RGB}{192,0,0}\n")
    lines.append("\definecolor{greenvec}{RGB}{0,192,0}\n")
    lines.append("\definecolor{bluevec}{RGB}{0,0,192}\n")
    lines.append("\definecolor{redvecl}{RGB}{192,164,164}\n")
    lines.append("\definecolor{greenvecl}{RGB}{164,192,164}\n")
    lines.append("\definecolor{bluevecl}{RGB}{164,164,192}\n")
    lines.append("\definecolor{orangevec}{RGB}{192,96,0}\n")
    # Start document and tikz
    lines.append("\\begin{document}\n")
    lines.append("    \\begin{tikzpicture}\n")
    # Border (clip)
    lines.append("        \\path[clip] (-1.15,-0.35) rectangle (7.67,5.10);\n")
    lines.append("        \\begin{axis}[boxplot/draw direction=y,\n")
    lines.append("                     height=177,\n")
    lines.append("                     width=262.5,\n")
    lines.append("                     ymajorgrids,\n")
    lines.append("                     xmin=0.25,\n")
    lines.append("                     xmax=%s,\n" % str(float(len(colnames))+0.75))
    if ylim[0] != None:
        lines.append("                     ymin=%s,\n" % str(ylim[0]))
    if ylim[1] != None:
        lines.append("                     ymax=%s,\n" % str(ylim[1]))
    lines.append("                     label style={font=\\footnotesize},\n")
    lines.append("                     tick label style={font=\\footnotesize,major tick length=0,/pgf/number format/fixed},\n")
    lines.append("                     ylabel={%s},\n" % ylabel)
    lines.append("                     legend entries={")
    for i in range(len(inst_labels)):
        lines[-1] += inst_labels[i]
        if i+1 != len(inst_labels):
            lines[-1] += ","
        else:
            lines[-1] += "},\n"
    lines.append("                     legend style={at={(0.5,1.0)},anchor=south,font=\\footnotesize,draw=none,fill=none,/tikz/every even column/.append style={column sep=20}},\n")
    lines.append("                     legend columns=%s,\n" % str(len(inst_labels)))
    lines.append("                     xtick={")
    lines.append("                     xticklabels={")
    lines.append("                     minor xtick={0.5,")
    for i in range(len(colnames)):
        lines[-3] += str(i+1)
        lines[-1] += str(float(i+1.5))
        if colnames[i] == "NO CR":
            lines[-2] += "\\,\\,"
        lines[-2] += colnames[i]
        if i+1 != len(colnames):
            lines[-3] += ","
            lines[-2] += ","
            lines[-1] += ","
    lines[-3] += "},\n"
    lines[-2] += "},\n"
    lines[-1] += "},\n"
    lines.append("                     minor grid style={dotted,lightgray},\n")
    lines.append("                     xminorgrids]\n")
    for i in range(len(inst_labels)):
        lines.append("            \\addlegendimage{legend image code/.code={\\draw[%s%s] (-0.02,-0.0125) rectangle (0.02,0.0125);\\draw[%s] (-0.04,-0.0125) -- (-0.04,0.0125);\\draw[%s] (0.04,-0.0125) -- (0.04,0.0125);\\draw[%s] (0.0,-0.0125) -- (0.0,0.0125);\\draw[%s] (-0.04,0.0) -- (-0.02,0.0);\\draw[%s] (0.04,0.0) -- (0.02,0.0);}}\n" %(colopt[i],filler[i],colopt[i],colopt[i],colopt[i],colopt[i],colopt[i]))
    for i in range(len(colnames)):
        for j in range(len(data[i])):
            lines.append("            \\addplot [boxplot prepared={\n")
            lines.append("                draw position  = %s,\n" % str(i+0.5+(j+1)*1./(len(data[i])+1)))
            lines.append("                upper whisker  = %s,\n" % str(data[i][j][6]))
            lines.append("                upper quartile = %s,\n" % str(data[i][j][4]))
            lines.append("                median         = %s,\n" % str(data[i][j][1]))
            lines.append("                lower quartile = %s,\n" % str(data[i][j][5]))
            lines.append("                lower whisker  = %s,\n" % str(data[i][j][7]))
            lines.append("                box extend     = %s\n"  % str(0.8/(len(data[i])+1)))
            lines.append("            }, style={very thin, draw=%s%s}] coordinates {};\n" % (colopt[j],filler[j]))

    lines.append("        \\end{axis}\n")
    lines.append("    \\end{tikzpicture}\n")
    lines.append("\\end{document}\n")
	

    # Write output
    scnFileName = os.path.join(outDir, "box_"+outName) 
    f           = open(scnFileName,"w")
    f.writelines(lines)
    f.close()

def boxplot3(colnames,data,inst_labels,ylabel,ylim,outDir,outName,fill=1,pref="box_"):
    if len(inst_labels) == 3:
        colopt = {2: "bluevec", 1: "greenvec", 0: "redvec"}
        if fill:
            filler = [", fill=redvecl",", fill=greenvecl",", fill=bluevecl"]
        else:
            filler = ["","",""]
    elif len(inst_labels) == 2:
        colopt = {1: "bluevec", 0: "greenvec"}
        if fill:
            filler = [", fill=greenvecl",", fill=bluevecl"]
        else:
            filler = ["",""]
    # Output
    lines = []
    
    # Preamble
#    lines.append("\\documentclass{standalone}\n")
#    lines.append("\\usepackage{xcolor}\n")
#    lines.append("\\usepackage{tikz}\n")
#    lines.append("\\usetikzlibrary{plotmarks}\n")
#    lines.append("\\usepackage{pgfplots}\n")
#    lines.append("\\pgfplotsset{compat=1.12}\usepgfplotslibrary{statistics}\n")
#    lines.append("\definecolor{redvec}{RGB}{192,0,0}\n")
#    lines.append("\definecolor{greenvec}{RGB}{0,192,0}\n")
#    lines.append("\definecolor{bluevec}{RGB}{0,0,192}\n")
#    lines.append("\definecolor{redvecl}{RGB}{192,164,164}\n")
#    lines.append("\definecolor{greenvecl}{RGB}{164,192,164}\n")
#    lines.append("\definecolor{bluevecl}{RGB}{164,164,192}\n")
#    lines.append("\definecolor{orangevec}{RGB}{192,96,0}\n")
#    # Start document and tikz
#    lines.append("\\begin{document}\n")
    lines.append("    \\begin{tikzpicture}\n")
    # Border (clip)
    lines.append("        \\path[clip] (-1.5,-0.4) rectangle (14.04,8.34);\n")
    lines.append("        \\begin{axis}[boxplot/draw direction=y,\n")
    lines.append("                     height=268,\n")
    lines.append("                     width=443.5,\n")
    lines.append("                     ymajorgrids,\n")
    lines.append("                     xmin=0.25,\n")
    lines.append("                     xmax=%s,\n" % str(float(len(colnames))+0.75))
    if ylim[0] != None:
        lines.append("                     ymin=%s,\n" % str(ylim[0]))
    if ylim[1] != None:
        lines.append("                     ymax=%s,\n" % str(ylim[1]))
    lines.append("                     tick label style={major tick length=0,/pgf/number format/fixed},\n")
    lines.append("                     ylabel={%s},\n" % ylabel)
    lines.append("                     legend entries={")
    for i in range(len(inst_labels)):
        lines[-1] += inst_labels[i]
        if i+1 != len(inst_labels):
            lines[-1] += ","
        else:
            lines[-1] += "},\n"
    lines.append("                     legend style={at={(0.5,1.0)},anchor=south,draw=none,fill=none,/tikz/every even column/.append style={column sep=20}},\n")
    lines.append("                     legend columns=%s,\n" % str(len(inst_labels)))
    lines.append("                     xtick={")
    lines.append("                     xticklabels={")
    lines.append("                     minor xtick={0.5,")
    for i in range(len(colnames)):
        lines[-3] += str(i+1)
        lines[-1] += str(float(i+1.5))
        if colnames[i] == "NO CR":
            lines[-2] += "\\,\\,"
        lines[-2] += colnames[i]
        if i+1 != len(colnames):
            lines[-3] += ","
            lines[-2] += ","
            lines[-1] += ","
    lines[-3] += "},\n"
    lines[-2] += "},\n"
    lines[-1] += "},\n"
    lines.append("                     minor grid style={dotted,lightgray},\n")
    lines.append("                     xminorgrids]\n")
    for i in range(len(inst_labels)):
        lines.append("            \\addlegendimage{legend image code/.code={\\draw[%s%s] (-0.02,-0.0125) rectangle (0.02,0.0125);\\draw[%s] (-0.04,-0.0125) -- (-0.04,0.0125);\\draw[%s] (0.04,-0.0125) -- (0.04,0.0125);\\draw[%s] (0.0,-0.0125) -- (0.0,0.0125);\\draw[%s] (-0.04,0.0) -- (-0.02,0.0);\\draw[%s] (0.04,0.0) -- (0.02,0.0);}}\n" %(colopt[i],filler[i],colopt[i],colopt[i],colopt[i],colopt[i],colopt[i]))
    for i in range(len(colnames)):
        for j in range(len(data[i])):
            lines.append("            \\addplot [boxplot prepared={\n")
            lines.append("                draw position  = %s,\n" % str(i+0.5+(j+1)*1./(len(data[i])+1)))
            lines.append("                upper whisker  = %s,\n" % str(data[i][j][6]))
            lines.append("                upper quartile = %s,\n" % str(data[i][j][4]))
            lines.append("                median         = %s,\n" % str(data[i][j][1]))
            lines.append("                lower quartile = %s,\n" % str(data[i][j][5]))
            lines.append("                lower whisker  = %s,\n" % str(data[i][j][7]))
            lines.append("                box extend     = %s\n"  % str(0.8/(len(data[i])+1)))
            lines.append("            }, style={very thin, draw=%s%s}] coordinates {};\n" % (colopt[j],filler[j]))

    lines.append("        \\end{axis}\n")
    lines.append("    \\end{tikzpicture}\n")
#    lines.append("\\end{document}\n")
	

    # Write output
    scnFileName = os.path.join(outDir, pref+outName) 
    f           = open(scnFileName,"w")
    f.writelines(lines)
    f.close()

# Settings
pdFileName    = 'final_thesis.pickle'
outDir        = 'C:\\Users\\suthe\\University\\Courses Year 5\\AE5310 - Thesis Control and Operations\\bluesky\\additional_tools\\post_processor\\plots\\'
outDir3       = 'C:\\Users\\suthe\\University\\Courses Year 5\\AE5310 - Thesis Control and Operations\\writing\\thesis\\plots\\'
finDir        = 'C:\\Users\\suthe\\University\\Courses Year 5\\AE5310 - Thesis Control and Operations\\writing\\scientific_paper\\tikz\\plots'
showTable     = False
showPlot      = False
showLatex     = True
buildmove     = True
out_cfl       = 'cfl.tex'
out_cfltime   = 'cfltime.tex'
out_timeincfl = 'timeincfl.tex'
out_los       = 'los.tex'
out_lostime   = 'lostime.tex'
out_timeinlos = 'timeinlos.tex'
out_dep       = 'dep.tex'
out_ipr       = 'ipr.tex'
out_sev       = 'sev.tex'
out_mcfl      = 'mcfl.tex'
out_time      = 'time.tex'
out_work      = 'work.tex'
out_density   = 'density.tex'
out_dist      = 'dist.tex'
out_timee     = 'timee.tex'
out_worke     = 'worke.tex'
out_densitye  = 'densitye.tex'
out_diste     = 'diste.tex'
out_inscmpcfl = 'inscmpcfl.tex'
out_inscmplos = 'inscmplos.tex'
Files = [out_cfl,out_cfltime,out_timeincfl,out_los,out_lostime,out_timeinlos,out_dep,out_ipr,out_sev,out_mcfl,out_time,out_work,out_density,out_dist,out_timee,out_worke,out_densitye,out_diste,out_inscmpcfl,out_inscmplos]

with open(pdFileName, 'rb') as handle:
    PD = pickle.load(handle)


insts   = np.unique(sorted(PD.inst))
vranges = np.unique(sorted(PD.vrange))
# Replace nans with 0
PD.lostime[PD.lostime != PD.lostime] = 0.0
#PD.numpify()

if showTable:
    for inst in insts:
        for vrange in vranges:
            idx = np.logical_and(PD.inst == inst, PD.vrange == vrange)
            colnames = np.unique(np.array(sorted(PD.method[idx])))
            rownames = np.array(["CFL","LOS","LOS_INTIME","AC with MCFL","MCFL_max","CFL/AC",\
                                 "TIME/AC","DIST/AC","WORK/AC","TIMEINCFL/CFL","TIMEINLOS/LOS","TIMEINCFL/AC","TIMEINLOS/AC","DENSITY"])
            show_data = np.zeros((rownames.shape[0],colnames.shape[0]))
            for i in range(show_data.shape[1]):
                ind = np.logical_and(PD.method == colnames[i],idx)
                N = float(sum(ind))
                show_data[0,i] = np.sum(PD.cfl_sum[ind])/N
                show_data[1,i] = np.sum(PD.los_sum[ind])/N
                show_data[2,i] = np.sum(PD.los_count[ind])/N
                show_data[3,i] = np.sum(PD.mcfl[ind])/N
                show_data[4,i] = np.sum(PD.mcfl_max[ind])/N
                show_data[5,i] = np.sum(PD.cfl_sum[ind]/PD.nac[ind])/N
                show_data[6,i] = np.sum(PD.time[ind])/N
                show_data[7,i] = np.sum(PD.dist[ind])/N
                show_data[8,i] = np.sum(PD.work[ind])/N
                show_data[9,i] = np.sum(PD.cfltime[ind])/N
                show_data[10,i] = np.sum(PD.lostime[ind])/N
                show_data[11,i] = np.sum(PD.cfltime[ind]*PD.cfl_sum[ind]/PD.nac[ind])/N
                show_data[12,i] = np.sum(PD.lostime[ind]*PD.los_sum[ind]/PD.nac[ind])/N
                show_data[13,i] = np.sum(PD.density[ind])/N
            
            pandas.set_option('expand_frame_repr', False)
    
            print '\n\033[4minst: ' + str(inst) + ' || ' + vrange + '\033[0m'
            print pandas.DataFrame(show_data,rownames,colnames)

if showPlot:
#    plt.close('all')
#    datas = []
    fig, axes = plt.subplots(nrows=5, ncols=len(insts))
    fig2, axes2 = plt.subplots(nrows=5, ncols=len(insts))
    fig.canvas.set_window_title(pdFileName[:-7] + ' - 1')
    fig2.canvas.set_window_title(pdFileName[:-7] + ' - 2')
    k = -1
    vrange = '450-500'
    for inst in insts:
        idx = np.logical_and(PD.inst == inst, PD.vrange == vrange)
        colnames = np.unique(np.array(sorted(PD.method[idx])))
        # Remove OFF-OFF        
        colnames2 = np.unique(np.array(sorted(PD.method[idx])))
        colnames2 = np.delete(colnames2,np.where(colnames2 == 'OFF-OFF')[0])
        
        boxes = []
        boxes2 = []
        boxes3 = []
        boxes4 = []
        boxes5 = []
        boxes6 = []
        boxes7 = []
        boxes8 = []
        boxes9 = []
        boxes10 = []
        for col in colnames:
            ind = np.logical_and(PD.method == col,idx)
            boxes.append(PD.cfl_sum[ind]/PD.nac[ind])
            boxes3.append(PD.time[ind])
            boxes4.append(PD.dist[ind])
            boxes5.append(PD.work[ind])
            boxes9.append(PD.density[ind])
        for col in colnames2:
            ind = np.logical_and(PD.method == col,idx)
            boxes2.append(PD.los_sum[ind]/PD.nac[ind])
            boxes6.append(PD.cfltime[ind])
            boxes7.append(PD.lostime[ind])
            boxes8.append(PD.mcfl[ind])
            boxes10.append(PD.los_count[ind]/PD.nac[ind])
        # Density
        dens = round(inst * 10000. / 455625.,1)
        k += 1
        if len(insts) > 1:
            axes[0,k].boxplot(boxes, labels=replaceColNames(colnames))
            axes[0,k].set_title('CFL/AC with ' + r'$\rho =$' + str(dens))
            axes[1,k].boxplot(boxes2, labels=replaceColNames(colnames2))
            axes[1,k].set_title('LOS/AC with ' + r'$\rho =$' + str(dens))
            axes[2,k].boxplot(boxes8, labels=replaceColNames(colnames2))
            axes[2,k].set_title('AC in MCFL with ' + r'$\rho =$' + str(dens))
            axes[3,k].boxplot(boxes9, labels=replaceColNames(colnames))
            axes[3,k].set_title('Density with ' + r'$\rho =$' + str(dens))
            axes[4,k].boxplot(boxes10, labels=replaceColNames(colnames2))
            axes[4,k].set_title('LOS/AC counted with ' + r'$\rho =$' + str(dens))
            axes2[0,k].boxplot(boxes3, labels=replaceColNames(colnames))
            axes2[0,k].set_title('Travel time with ' + r'$\rho =$' + str(dens))
            axes2[1,k].boxplot(boxes4, labels=replaceColNames(colnames))
            axes2[1,k].set_title('Travel dist with' + r'$\rho =$' + str(dens))
            axes2[2,k].boxplot(boxes5, labels=replaceColNames(colnames))
            axes2[2,k].set_title('Work done with ' + r'$\rho =$' + str(dens))
            axes2[3,k].boxplot(boxes6, labels=replaceColNames(colnames2))
            axes2[3,k].set_title('Time in CFL with ' + r'$\rho =$' + str(dens))
            axes2[4,k].boxplot(boxes7, labels=replaceColNames(colnames2))
            axes2[4,k].set_title('Time in LOS with ' + r'$\rho =$' + str(dens))
        else:
            axes[0].boxplot(boxes, labels=replaceColNames(colnames))
            axes[0].set_title('CFL/AC with ' + r'$\rho =$' + str(dens))
            axes[1].boxplot(boxes2, labels=replaceColNames(colnames2))
            axes[1].set_title('LOS/AC with ' + r'$\rho =$' + str(dens))
            axes[2].boxplot(boxes8, labels=replaceColNames(colnames2))
            axes[2].set_title('AC in MCFL with ' + r'$\rho =$' + str(dens))
            axes[3].boxplot(boxes9, labels=replaceColNames(colnames))
            axes[3].set_title('Density with ' + r'$\rho =$' + str(dens))
            axes[4].boxplot(boxes10, labels=replaceColNames(colnames2))
            axes[4].set_title('LOS/AC counted with ' + r'$\rho =$' + str(dens))
            axes2[0].boxplot(boxes3, labels=replaceColNames(colnames))
            axes2[0].set_title('Travel time with ' + r'$\rho =$' + str(dens))
            axes2[1].boxplot(boxes4, labels=replaceColNames(colnames))
            axes2[1].set_title('Travel dist with' + r'$\rho =$' + str(dens))
            axes2[2].boxplot(boxes5, labels=replaceColNames(colnames))
            axes2[2].set_title('Work done with ' + r'$\rho =$' + str(dens))
            axes2[3].boxplot(boxes6, labels=replaceColNames(colnames2))
            axes2[3].set_title('Time in CFL with ' + r'$\rho =$' + str(dens))
            axes2[4].boxplot(boxes7, labels=replaceColNames(colnames2))
            axes2[4].set_title('Time in LOS with ' + r'$\rho =$' + str(dens))
#        fig = plt.figure()
#        plt.boxplot(boxes)
#    for i in range(len(insts)):
#    latexify()
    plt.show()

if showLatex:
    # Progress
    print "Writing .tex-files"
#    fig1 = plt.figure()
    conf     = 99
    vrange   = '450-500'
    colnames = ["MVP","RS1","RS2","RS3","RS4","RS5","RS6","RS7","RS8","NO CR"]
    inst_labels = ["Low","Moderate","High"]
    insts    = [113,227,341]
    ylabel_cfl       = "Number of Conflicts Per Flight [-]"
    ylabel_cfltime   = "Duration of Conflicts [s]"
    ylabel_timeincfl = "Time in Conflicts Per Flight [s]"
    ylabel_los       = "Number of Intrusions Per Flight [-]"
    ylabel_lostime   = "Duration of Intrusions [s]"
    ylabel_timeinlos = "Time in LoS Per Flight [s]"
    ylabel_dep       = "Domino Effect Parameter [-]"
    ylabel_ipr       = "Intrusion Prevention Rate [\\%]"
    ylabel_sev       = "Intrusion Severity [-]"
    ylabel_mcfl      = "Portion of AC with Multi-AC Conflicts [\\%]"
    ylabel_time      = "Total Flight Time [h]"
    ylabel_work      = "Work Done [GJ]"
    ylabel_density   = "Density [AC/10,000\\,NM\\textsuperscript{2}]"
    ylabel_dist      = "Travel Distance [NM]"
    ylabel_timee     = "Extra Travel Time [\\%]"
    ylabel_worke     = "Extra Work Done [\\%]"
    ylabel_densitye  = "Extra Density [\\%]"
    ylabel_diste     = "Extra Travel Distance [\\%]"
    ylabel_inscmpcfl = "Extra Conflicts [\\%]"
    ylabel_inscmplos = "Extra Intrusions [\\%]"
#    if len(insts) == 1:
#        colopt = {0: "black"}
#    elif len(insts) == 2:
#        colopt = {1: "black", 0: "black!25"}
#    elif len(insts) == 3:
##        colopt = {2: "black", 1: "black!62", 0: "black!25"}
#        colopt = {2: "bluevec", 1: "greenvec", 0: "redvec"}
##\definecolor{redvec}{RGB}{192,0,0}
##\definecolor{greenvec}{RGB}{0,192,0}
##\definecolor{bluevec}{RGB}{0,0,192}
##\definecolor{orangevec}{RGB}{192,96,0}
#    elif len(insts) == 4:
#        colopt = {3: "black", 2: "black!75", 1: "black!50", 0: "black!25"}
#    box_cfl = []
    dat_cfl = []
    dat_cfltime = []
    dat_timeincfl = []
    dat_los = []
    dat_lostime = []
    dat_timeinlos = []
    dat_dep = []
    dat_ipr = []
    dat_sev = []
    dat_mcfl = []
    dat_time = []
    dat_work = []
    dat_density = []
    dat_dist = []
    dat_timee = []
    dat_worke = []
    dat_densitye = []
    dat_diste = []
    dat_inscmpcfl = []
    dat_inscmplos = []

    dat3_cfl = []
    dat3_cfltime = []
    dat3_timeincfl = []
    dat3_los = []
    dat3_lostime = []
    dat3_timeinlos = []
    dat3_dep = []
    dat3_ipr = []
    dat3_sev = []
    dat3_mcfl = []
    dat3_time = []
    dat3_work = []
    dat3_density = []
    dat3_dist = []
    dat3_timee = []
    dat3_worke = []
    dat3_densitye = []
    dat3_diste = []
    k = -1        
    for col in colnames:
        idx = np.logical_and(PD.method == col, PD.vrange == vrange)
        idx3 = np.logical_and(PD.method == col, PD.vrange == '300-500')
        k += 1
#        box_cfl.append([])
        dat_cfl.append([])
        dat_cfltime.append([])
        dat_timeincfl.append([])
        dat_los.append([])
        dat_lostime.append([])
        dat_timeinlos.append([])
        dat_dep.append([])
        dat_ipr.append([])
        dat_sev.append([])
        dat_mcfl.append([])
        dat_time.append([])
        dat_work.append([])
        dat_density.append([])
        dat_dist.append([])
        dat_timee.append([])
        dat_worke.append([])
        dat_densitye.append([])
        dat_diste.append([])
        dat3_cfl.append([])
        dat3_cfltime.append([])
        dat3_timeincfl.append([])
        dat3_los.append([])
        dat3_lostime.append([])
        dat3_timeinlos.append([])
        dat3_dep.append([])
        dat3_ipr.append([])
        dat3_sev.append([])
        dat3_mcfl.append([])
        dat3_time.append([])
        dat3_work.append([])
        dat3_density.append([])
        dat3_dist.append([])
        dat3_timee.append([])
        dat3_worke.append([])
        dat3_densitye.append([])
        dat3_diste.append([])
        for inst in insts:
            ind = np.logical_and(PD.inst == inst,idx)
            entry_cfl = PD.cfl_sum[ind]/PD.nac[ind]
            entry_cfltime = PD.cfltime[ind]
            entry_timeincfl = PD.timeincfl[ind]
            entry_los = PD.los_sum[ind]/PD.nac[ind]
            entry_lostime = PD.lostime[ind]
            entry_timeinlos = PD.timeinlos[ind]
            entry_dep = PD.dep[ind]
            entry_ipr = PD.ipr[ind] * 100
            entry_sev = PD.severities[ind]
            entry_mcfl = PD.mcfl[ind]/PD.nac[ind] * 100
            entry_time = PD.time[ind] / 3600
            entry_work = PD.work[ind] / 1e9
            entry_density = PD.density[ind]
            entry_dist = PD.dist[ind] / 1852
            entry_timee = PD.timee[ind] * 100
            entry_worke = PD.worke[ind] * 100
            entry_densitye = PD.densitye[ind] * 100
            entry_diste = PD.diste[ind] * 100
            ind3 = np.logical_and(PD.inst == inst,idx3)
            entry3_cfl = PD.cfl_sum[ind3]/PD.nac[ind3]
            entry3_cfltime = PD.cfltime[ind3]
            entry3_timeincfl = PD.timeincfl[ind3]
            entry3_los = PD.los_sum[ind3]/PD.nac[ind3]
            entry3_lostime = PD.lostime[ind3]
            entry3_timeinlos = PD.timeinlos[ind3]
            entry3_dep = PD.dep[ind3]
            entry3_ipr = PD.ipr[ind3] * 100
            entry3_sev = PD.severities[ind3]
            entry3_mcfl = PD.mcfl[ind3]/PD.nac[ind3] * 100
            entry3_time = PD.time[ind3] / 3600
            entry3_work = PD.work[ind3] / 1e9
            entry3_density = PD.density[ind3]
            entry3_dist = PD.dist[ind3] / 1852
            entry3_timee = PD.timee[ind3] * 100
            entry3_worke = PD.worke[ind3] * 100
            entry3_densitye = PD.densitye[ind3] * 100
            entry3_diste = PD.diste[ind3] * 100
#            box_cfl[k].append(entry)
            dat_cfl[k].append(stats(entry_cfl, conf))
            dat_cfltime[k].append(stats(entry_cfltime, conf))
            dat_timeincfl[k].append(stats(entry_timeincfl, conf))
            dat_los[k].append(stats(entry_los, conf))
            dat_lostime[k].append(stats(entry_lostime, conf))
            dat_timeinlos[k].append(stats(entry_timeinlos, conf))
            dat_dep[k].append(stats(entry_dep, conf))
            dat_ipr[k].append(stats(entry_ipr, conf))
            dat_sev[k].append(stats(entry_sev, conf))
            dat_mcfl[k].append(stats(entry_mcfl, conf))
            dat_time[k].append(stats(entry_time, conf))
            dat_work[k].append(stats(entry_work, conf))
            dat_density[k].append(stats(entry_density, conf))
            dat_dist[k].append(stats(entry_dist, conf))
            dat_timee[k].append(stats(entry_timee, conf))
            dat_worke[k].append(stats(entry_worke, conf))
            dat_densitye[k].append(stats(entry_densitye, conf))
            dat_diste[k].append(stats(entry_diste, conf))
            
            dat3_cfl[k].append(stats(entry3_cfl, conf))
            dat3_cfltime[k].append(stats(entry3_cfltime, conf))
            dat3_timeincfl[k].append(stats(entry3_timeincfl, conf))
            dat3_los[k].append(stats(entry3_los, conf))
            dat3_lostime[k].append(stats(entry3_lostime, conf))
            dat3_timeinlos[k].append(stats(entry3_timeinlos, conf))
            dat3_dep[k].append(stats(entry3_dep, conf))
            dat3_ipr[k].append(stats(entry3_ipr, conf))
            dat3_sev[k].append(stats(entry3_sev, conf))
            dat3_mcfl[k].append(stats(entry3_mcfl, conf))
            dat3_time[k].append(stats(entry3_time, conf))
            dat3_work[k].append(stats(entry3_work, conf))
            dat3_density[k].append(stats(entry3_density, conf))
            dat3_dist[k].append(stats(entry3_dist, conf))
            dat3_timee[k].append(stats(entry3_timee, conf))
            dat3_worke[k].append(stats(entry3_worke, conf))
            dat3_densitye[k].append(stats(entry3_densitye, conf))
            dat3_diste[k].append(stats(entry3_diste, conf))
    k = -1      
    for col in colnames:
        k += 1
        dat_inscmpcfl.append([])
        dat_inscmplos.append([])
        for inst in insts[1:]:
            idx = np.where(np.logical_and(np.logical_and(PD.method == col, PD.vrange == '300-500'), PD.inst == inst))[0]
            entry_inscmpcfl = np.zeros(len(idx), dtype=np.float32)
            entry_inscmplos = np.zeros(len(idx), dtype=np.float32)
            l = -1
            for j in idx:
                l += 1
                i = np.where(np.logical_and(np.logical_and(np.logical_and(PD.method == col, PD.vrange == vrange), PD.inst == inst), PD.rep == PD.rep[j]))[0][0]
                entry_inscmpcfl[l] = (float(PD.cfl_sum[i]) -  float(PD.cfl_sum[j]))/max(PD.cfl_sum[j],1) * 100
                entry_inscmplos[l] = (float(PD.los_sum[i]) -  float(PD.los_sum[j]))/max(PD.los_sum[j],1) * 100
            dat_inscmpcfl[k].append(stats(entry_inscmpcfl, conf))
            dat_inscmplos[k].append(stats(entry_inscmplos, conf))

    confplot(colnames,dat_cfl,inst_labels,ylabel_cfl,[0,None],outDir,out_cfl)
    confplot(colnames[:-1],dat_cfltime,inst_labels,ylabel_cfltime,[0,None],outDir,out_cfltime)
    confplot(colnames[:-1],dat_timeincfl,inst_labels,ylabel_timeincfl,[0,150],outDir,out_timeincfl)
    confplot(colnames[:-1],dat_los,inst_labels,ylabel_los,[0,0.2],outDir,out_los)
    confplot(colnames[:-1],dat_lostime,inst_labels,ylabel_lostime,[0,None],outDir,out_lostime)
    confplot(colnames[:-1],dat_timeinlos,inst_labels,ylabel_timeinlos,[0,10],outDir,out_timeinlos)
    confplot(colnames[:-1],dat_dep,inst_labels,ylabel_dep,[0,None],outDir,out_dep)
    confplot(colnames[:-1],dat_ipr,inst_labels,ylabel_ipr,[92,100],outDir,out_ipr)
    confplot(colnames[:-1],dat_sev,inst_labels,ylabel_sev,[0,None],outDir,out_sev)
    confplot(colnames[:-1],dat_mcfl,inst_labels,ylabel_mcfl,[0,None],outDir,out_mcfl)
    confplot(colnames,dat_time,inst_labels,ylabel_time,[0.4,None],outDir,out_time)
    confplot(colnames,dat_work,inst_labels,ylabel_work,[60,None],outDir,out_work)
    confplot(colnames,dat_density,inst_labels,ylabel_density,[0,None],outDir,out_density)
    confplot(colnames,dat_dist,inst_labels,ylabel_dist,[200,None],outDir,out_dist)
    confplot(colnames[:-1],dat_timee,inst_labels,ylabel_timee,[0,None],outDir,out_timee)
    confplot(colnames[:-1],dat_worke,inst_labels,ylabel_worke,[0,None],outDir,out_worke)
    confplot(colnames[:-1],dat_densitye,inst_labels,ylabel_densitye,[0,None],outDir,out_densitye)
    confplot(colnames[:-1],dat_diste,inst_labels,ylabel_diste,[0,None],outDir,out_diste)
    confplot(colnames[:-1],dat_inscmpcfl,inst_labels[1:],ylabel_inscmpcfl,[-20,40],outDir,out_inscmpcfl)
    confplot(colnames[:-1],dat_inscmplos,inst_labels[1:],ylabel_inscmplos,[-50,100],outDir,out_inscmplos)
    
    boxplot(colnames,dat_cfl,inst_labels,ylabel_cfl,[0,None],outDir,out_cfl,1)
    boxplot(colnames[:-1],dat_cfltime,inst_labels,ylabel_cfltime,[0,None],outDir,out_cfltime)
    boxplot(colnames[:-1],dat_timeincfl,inst_labels,ylabel_timeincfl,[0,150],outDir,out_timeincfl)
    boxplot(colnames[:-1],dat_los,inst_labels,ylabel_los,[0,0.2],outDir,out_los)
    boxplot(colnames[:-1],dat_lostime,inst_labels,ylabel_lostime,[0,None],outDir,out_lostime)
    boxplot(colnames[:-1],dat_timeinlos,inst_labels,ylabel_timeinlos,[0,10],outDir,out_timeinlos)
    boxplot(colnames[:-1],dat_dep,inst_labels,ylabel_dep,[0,None],outDir,out_dep)
    boxplot(colnames[:-1],dat_ipr,inst_labels,ylabel_ipr,[92,100],outDir,out_ipr)
    boxplot(colnames[:-1],dat_sev,inst_labels,ylabel_sev,[0,None],outDir,out_sev)
    boxplot(colnames[:-1],dat_mcfl,inst_labels,ylabel_mcfl,[0,None],outDir,out_mcfl)
    boxplot(colnames,dat_time,inst_labels,ylabel_time,[0.4,None],outDir,out_time)
    boxplot(colnames,dat_work,inst_labels,ylabel_work,[60,None],outDir,out_work)
    boxplot(colnames,dat_density,inst_labels,ylabel_density,[0,None],outDir,out_density)
    boxplot(colnames,dat_dist,inst_labels,ylabel_dist,[200,None],outDir,out_dist)
    boxplot(colnames[:-1],dat_timee,inst_labels,ylabel_timee,[0,None],outDir,out_timee)
    boxplot(colnames[:-1],dat_worke,inst_labels,ylabel_worke,[0,None],outDir,out_worke)
    boxplot(colnames[:-1],dat_densitye,inst_labels,ylabel_densitye,[0,None],outDir,out_densitye)
    boxplot(colnames[:-1],dat_diste,inst_labels,ylabel_diste,[0,None],outDir,out_diste)
    boxplot(colnames[:-1],dat_inscmpcfl,inst_labels[1:],ylabel_inscmpcfl,[-20,40],outDir,out_inscmpcfl)
    boxplot(colnames[:-1],dat_inscmplos,inst_labels[1:],ylabel_inscmplos,[-50,100],outDir,out_inscmplos)

    boxplot3(colnames,dat3_cfl,inst_labels,ylabel_cfl,[0,None],outDir3,out_cfl,1,"box3_")
    boxplot3(colnames[:-1],dat3_cfltime,inst_labels,ylabel_cfltime,[0,None],outDir3,out_cfltime,1,"box3_")
    boxplot3(colnames[:-1],dat3_timeincfl,inst_labels,ylabel_timeincfl,[0,150],outDir3,out_timeincfl,1,"box3_")
    boxplot3(colnames[:-1],dat3_los,inst_labels,ylabel_los,[0,0.2],outDir3,out_los,1,"box3_")
    boxplot3(colnames[:-1],dat3_lostime,inst_labels,ylabel_lostime,[0,None],outDir3,out_lostime,1,"box3_")
    boxplot3(colnames[:-1],dat3_timeinlos,inst_labels,ylabel_timeinlos,[0,10],outDir3,out_timeinlos,1,"box3_")
    boxplot3(colnames[:-1],dat3_dep,inst_labels,ylabel_dep,[0,None],outDir3,out_dep,1,"box3_")
    boxplot3(colnames[:-1],dat3_ipr,inst_labels,ylabel_ipr,[92,100],outDir3,out_ipr,1,"box3_")
    boxplot3(colnames[:-1],dat3_sev,inst_labels,ylabel_sev,[0,None],outDir3,out_sev,1,"box3_")
    boxplot3(colnames[:-1],dat3_mcfl,inst_labels,ylabel_mcfl,[0,None],outDir3,out_mcfl,1,"box3_")
    boxplot3(colnames,dat3_time,inst_labels,ylabel_time,[0.4,None],outDir3,out_time,1,"box3_")
    boxplot3(colnames,dat3_work,inst_labels,ylabel_work,[60,None],outDir3,out_work,1,"box3_")
    boxplot3(colnames,dat3_density,inst_labels,ylabel_density,[0,None],outDir3,out_density,1,"box3_")
    boxplot3(colnames,dat3_dist,inst_labels,ylabel_dist,[200,None],outDir3,out_dist,1,"box3_")
    boxplot3(colnames[:-1],dat3_timee,inst_labels,ylabel_timee,[0,None],outDir3,out_timee,1,"box3_")
    boxplot3(colnames[:-1],dat3_worke,inst_labels,ylabel_worke,[0,None],outDir3,out_worke,1,"box3_")
    boxplot3(colnames[:-1],dat3_densitye,inst_labels,ylabel_densitye,[0,None],outDir3,out_densitye,1,"box3_")
    boxplot3(colnames[:-1],dat3_diste,inst_labels,ylabel_diste,[0,None],outDir3,out_diste,1,"box3_")

    if buildmove:
        # Progress
        print "Writing .pdf-files"
        for File in Files:
            os.system('pdflatex -quiet -output-directory=./plots/ ./plots/'+'bar_'+File)
            os.system('pdflatex -quiet -output-directory=./plots/ ./plots/'+'box_'+File)
        # Progress
        print "Moving .tex- and .pdf-files"
        for File in Files:
            os.system('copy "'+outDir+'bar_'+File+'" "'+finDir+'" >nul')
            os.system('copy "'+outDir+'bar_'+File[:-4]+'.pdf" "'+finDir+'" >nul')
            os.system('copy "'+outDir+'box_'+File+'" "'+finDir+'" >nul')
            os.system('copy "'+outDir+'box_'+File[:-4]+'.pdf" "'+finDir+'" >nul')
    # Progress
    print "Done."

#    # Output
#    lines = []
#    
#    # Preamble
#    lines.append("\\documentclass{standalone}\n")
#    lines.append("\\usepackage{xcolor}\n")
#    lines.append("\\usepackage{tikz}\n")
#    lines.append("\\usetikzlibrary{plotmarks}\n")
#    lines.append("\\usepackage{pgfplots}\n")
#    lines.append("\\pgfplotsset{compat=1.12}\usepgfplotslibrary{statistics}\n")
#    lines.append("\\renewcommand{\\sfdefault}{phv}\n")
#    lines.append("\\renewcommand{\\rmdefault}{ptm}\n")
#    lines.append("\\renewcommand{\\ttdefault}{pcr}\n")
#    # Start document and tikz
#    lines.append("\\begin{document}\n")
#    lines.append("    \\begin{tikzpicture}\n")
#    # Border (clip)
#    lines.append("        \\path[clip] (-0.93,-0.33) rectangle (7.89,4.63);\n")
#    lines.append("        \\begin{axis}[height=165,\n")
#    lines.append("                     width=269.5,\n")
#    lines.append("                     enlarge y limits,\n")
#    lines.append("                     ymajorgrids,\n")
#    lines.append("                     ymin=1,\n")
#    lines.append("                     label style={font=\\footnotesize},\n")
#    lines.append("                     tick label style={font=\\footnotesize},\n")
#    lines.append("                     ylabel={Number of Conflicts Per Flight},\n")
#    lines.append("                     legend style={at={(0.5,1.0)},anchor=south,font=\\footnotesize,draw=none,fill=none,/tikz/every even column/.append style={column sep=20}},\n")
#    lines.append("                     legend columns=3,\n")
#    lines.append("                     xtick={")
#    lines.append("                     xticklabels={")
#    for i in range(len(colnames)):
#        lines[-2] += str(i+1)
#        if colnames[i] == "NO CR":
#            lines[-1] += "\\,\\,"
#        lines[-1] += colnames[i]
#        if i+1 != len(colnames):
#            lines[-2] += ","
#            lines[-1] += ","
#    lines[-2] += "},\n"
#    lines[-1] += "}]\n"
#
#    for i in range(len(dat_cfl)):
#        for j in range(len(dat_cfl[i])):
#            lines.append("            \\addplot [only marks, %s, fill=%s, draw=%s, error bars/.cd, y dir=both, y explicit, error mark options={%s, rotate=90, mark size=2}] coordinates {\n" % (markopt[j],colopt[j],colopt[j],colopt[j]))
#            lines.append("                (%s,%s) += (0,%s) -= (0,%s)};\n" % (str(i+0.5+(j+1)*1./(len(dat_cfl[i])+1)), str(dat_cfl[i][j][0]), str(dat_cfl[i][j][8]-dat_cfl[i][j][0]), str(dat_cfl[i][j][0]-dat_cfl[i][j][9]))) 
#
#    lines.append("            \\legend{")
#    for i in range(len(inst_labels)):
#        lines[-1] += inst_labels[i]
#        if i+1 != len(inst_labels):
#            lines[-1] += ","
#        else:
#            lines[-1] += "}\n"
#    lines.append("        \\end{axis}\n")
#    lines.append("    \\end{tikzpicture}\n")
#    lines.append("\\end{document}\n")
#	
#
#    # Write output
#    scnFileName = os.path.join(outDir, outName) 
#    f           = open(scnFileName,"w")
#    f.writelines(lines)
#    f.close()
    
    
    
#    if 0:
#        lines.append("        \\path[clip] (-0.95,-0.33) rectangle (7.87,5.12);\n")
#        lines.append("        \\begin{axis}[boxplot/draw direction=y,\n")
#        lines.append("                     height=190,\n")
#        lines.append("                     width=267,\n")
#        lines.append("                     enlarge y limits,\n")
#        lines.append("                     ymajorgrids,\n")
#        lines.append("                     label style={font=\\footnotesize},\n")
#        lines.append("                     tick label style={font=\\footnotesize},\n")
#        lines.append("                     ylabel={Number of Conflicts Per Flight},\n")
#        lines.append("                     xtick={")
#        lines.append("                     xticklabels={")
#        for i in range(len(colnames)):
#            lines[-2] += str(i+1)
#            if colnames[i] == "NO CR":
#                lines[-1] += "\\,\\,"
#            lines[-1] += colnames[i]
#            if i+1 != len(colnames):
#                lines[-2] += ","
#                lines[-1] += ","
#        lines[-2] += "},\n"
#        lines[-1] += "}]\n"
#        
#        for i in range(len(dat_cfl)):
#            for j in range(len(dat_cfl[i])):
#                lines.append("            \\addplot [boxplot prepared={\n")
#                lines.append("                draw position  = %s,\n" % str(i+0.5+(j+1)*1./(len(dat_cfl[i])+1)))
#                lines.append("                upper whisker  = %s,\n" % str(dat_cfl[i][j][6]))
#                lines.append("                upper quartile = %s,\n" % str(dat_cfl[i][j][4]))
#                lines.append("                median         = %s,\n" % str(dat_cfl[i][j][1]))
#                lines.append("                lower quartile = %s,\n" % str(dat_cfl[i][j][5]))
#                lines.append("                lower whisker  = %s,\n" % str(dat_cfl[i][j][7]))
#                lines.append("                box extend     = %s\n"  % str(0.8/(len(dat_cfl[i])+1)))
#                lines.append("            }] coordinates {};\n")
#    else: