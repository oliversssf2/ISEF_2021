import sys
import os
import csv

if __name__ == "__main__":
    currentdir = os.getcwd()
    name = sys.argv[2]
    new_dir = os.path.join(currentdir, name)
    print(new_dir)

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)   
    new_file = os.path.join(new_dir,name)+'.csv'
    
    with open (new_file, mode='w+', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow("HHHHHHH")
    