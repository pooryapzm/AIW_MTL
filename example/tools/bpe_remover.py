import glob, os, sys

in_fname=sys.argv[1]
out_fname =sys.argv[2]

out_file = open(out_fname,'w')
with open(in_fname) as file:
    for raw_line in file:
        splitted_line = raw_line.split()
        new_line=""
        for i in range(0,len(splitted_line)):
            if splitted_line[i].endswith("@@"):
                new_line += splitted_line[i][:-2]
            else:
                new_line += splitted_line[i]+" "
        out_file.write(new_line.strip()+"\n")
