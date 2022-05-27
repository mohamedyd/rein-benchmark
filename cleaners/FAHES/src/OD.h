/**************************************
 **** 2017-6-5      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/
/****************************************************************************
 -- num_non_ignorable_strings: if the column contains more than num_non_ignorable_strings
 	then outlier detection is applicable
-- num_od_tools: the number of outlier detection tools that will be used
-- sq: the exponent that will be used in the pow() function
-- least_num_values: is the least number of distinct values to be able to run 
	outlier detection
****************************************************************************/
#ifndef _Outlier_Detector_
#define _Outlier_Detector_



#include "Table.h"
#include "common.h"

class OD {
	int num_non_ignorable_strings = 3;		 
	int num_od_tools = 1;
	int EX = 2;
	int least_num_values = 10;
public:
    OD(){ }
    
    // double z_score(double val, const double & mean, const double & std){
    // 	double z_sc = (fabs(val - mean)) / std;
    // 	return z_sc;
    // }
    void compute_statistical_quantities(map<double, long> & col_profile, double & mean, 
    									double & std);
    void detect_outliers(TableProfile & TP, vector<sus_disguised> & sus_dis_values);
    // vector<sus_disguised> z_score_od(const string Attribute, map<double, long> & col_profile, 
    // 									const double & mean, const double & std);
};

#endif