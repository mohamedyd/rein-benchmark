/**************************************
 **** 2017-6-3      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#ifndef _Detector_
#define _Detector_



#include "Table.h"
#include "common.h"

class DV_Detector {
public:
    DV_Detector(){ }
    void detect_single_char_strings(TableProfile TP, 
    			std::vector<sus_disguised> & sus_dis_values);
    void positive_negative_inconsistency(TableProfile TP, 
    			std::vector<sus_disguised> & sus_dis_values);
    void check_repeated_substrings(TableProfile TP, 
                vector<map<string, long> > & M,
    			std::vector<sus_disguised> & sus_dis_values);
    void check_non_conforming_patterns(TableProfile & TP,
                vector<map<string, long> > & M,  
    			std::vector<sus_disguised> & sus_dis_values);
    // vector< sus_disguised > Mod_DiMaC(const Table & T, TableProfile & TP);
};

#endif 