/**************************************
 **** 2017-3-6      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/


#ifndef _Profiler_
#define _Profiler_



#include "Table.h"
#include "common.h"
#include "csv_reader.h"

class DataProfiler {
public:
    DataProfiler(){ }
    vector<map<string, long> > TableHistogram(const Table & T);
    vector< sus_disguised > find_disguised_values(const Table & T);
    float compare_distribution(const long , const long, const long, 
    							vector<map<string, long> > &, 
    							vector<map<string, long> > &);
    // bool prune_value(const string &, const vector<map<string, long> > &, const vector<map<string, long> > &);
    bool prune_attribute(const long, const long, const vector<map<string, long> > &);
    void PrintTableHist(vector<map<string, long> > m_tablehist);
    long find_least_distinct_values(vector<map<string, long> > tabhist);
    TableProfile profile_table(const Table & T, vector<map<string, long> > & TabHist);
};
#endif