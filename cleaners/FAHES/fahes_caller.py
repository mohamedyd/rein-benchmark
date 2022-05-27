import os
from subprocess import Popen
import json 
from os import listdir
import ctypes
from ctypes import c_char_p
import sys

global tool_loc
tool_loc = "./src/"



def executeFahes(input_table, tool_id): 
    """
    Method that execute Fahes.

    Arguments:
    input_table (String) -- Path to the input table for which fahes should be executed
    tool_id (int) -- which fahes component is to be run. 1 = check syntactic outliers only);
                     (2 = detect DMVs that replace MAR values); (3 = detect DMVs that are numerical outliers only);
                     (4 = check all DMVs)

    Returns:
    dmvs_table_name (String) -- Path to the .csv that holds the results of fahes
    """
    out_dir = 'Results/'
    
    output_dir = ""
    if out_dir:
        output_dir = os.path.join(os.path.dirname(__file__), out_dir)
        #output_dir = os.path.join(os.getcwd(), out_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tName = os.path.join(os.getcwd(), input_table)
    callFahes(input_table, output_dir, tool_id)
    tab_name = input_table[input_table.rindex('/')+1: len(input_table)]
    #tab_name = tab_name.replace('.csv', '.json')
    dmvs_table_name = os.path.join(output_dir, 'DMV_' + tab_name)
    return dmvs_table_name
                    
        
        

def callFahes(tab_full_name, output_dir, tool_id):
    global tool_loc
   
    g_path = os.path.join(os.path.dirname(__file__), "src")

    libfile = ""
    if not g_path.endswith('/'):
        libfile = g_path + "/libFahes.so"
    else:
        libfile = g_path + "libFahes.so"

    tab_name = c_char_p(tab_full_name.encode('utf-8'))
    out_dir = c_char_p(output_dir.encode('utf-8'))

    # open shared lib
    Fahes = ctypes.CDLL(libfile)

    # tell python parameter funktioniert so!
    Fahes.execute.restype = None
    Fahes.execute.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    Fahes.execute(tab_name, out_dir, tool_id)
    

def main():
    # check arguments
    # for arg in sys.argv[1:]:
    if(len(sys.argv) != 2):
        print("Wrong number of arguments .. entered (",len(sys.argv),")")
        # print(sys.argv, file=sys.stderr)
        print("Usage (",sys.argv[0],"): <data file name>")
        sys.exit(1)

    table_name = os.path.abspath(sys.argv[1]);
    
    # print(table_name)
    executeFahes(table_name)
    



if __name__ == "__main__":
    main()
