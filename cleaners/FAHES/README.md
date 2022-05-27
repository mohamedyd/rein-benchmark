## FAHES
#### FAHES which took its name from the Arabic word that means inspector, is a tool for detecting the disguised missing values. By disguised missing values, we refer to the values in a table that replaces the missing values.
#### FAHES is a service integrated with the data civilizer system that can work under the Civilizer Studio or it can work as a stand-alone tool.
#### To compile FAHES, change to directory src/ and simply type make from the terminal. 
$ make
#### This will produce an executable file called "FAHES"

### FAHES requires the following arguments:
#### The name of the data file: FAHES currently works on csv data formats.

#### Output directory name: specifies the place where to put the output table.

#### Tool id: gives the user flexibility to run specific component of FAHES. (1 = check syntactic outliers only), (2 = detect DMVs that replace MAR values), (3 = detect DMVs that are numerical outliers only) and (4 = check all DMVs)

The output tables contain meta-data that describes the disguised missing values. Each file contain four columns:
Table Name
Column Name 
The value that detected as disguised missing value
The frequency (the number of occurrences of the value in that column)
The tool that detected this DMV. 

#### To run FAHES (example):
$ cd FAHES/src/

$ make clean && make

$ ./FAHES ../Data/323-1.csv ../Results/ 4 

This command will detect all detectable DMVs.
