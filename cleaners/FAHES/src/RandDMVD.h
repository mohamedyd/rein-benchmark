/**************************************
 **** 2017-7-25      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/


#ifndef _RandDMVD_
#define _RandDMVD_



#include "Table.h"
#include "common.h"
// #include "csv_reader.h"
typedef vector<map <string, doubleVecStr> > RandDMVD_Index;
class RandDMVD {
private:
    RandDMVD_Index RandDMVD_Index_T;
public:
    RandDMVD();
    RandDMVD(const Table & T) {  Table_Index_RandDMVD(T, RandDMVD_Index_T);  }
    void Table_Index_RandDMVD(const Table & T, RandDMVD_Index & IDX);
    // Table SELECT(const Table & T, const string & item, const long & idx);
    double subtable_correlation(const Table & , const string,
                                const long , long & );
    bool prune_attribute(const long idx, vector<map<string, long> > & M);
    vector< sus_disguised > find_disguised_values(const Table & T,
                                    vector<map<string, long> > & tablehist,
                                    long max_num_terms_per_att);
    long compute_num_cooccur(const vector<string> &, const long &);
    long compute_num_occur(string v, long A);
    long compute_num_cooccur(const vector<string> &, const long &, RandDMVD_Index &);
    long compute_num_occur(string v, long A, RandDMVD_Index & RandDMVD_Index_subT);
    double record_correlation(const vector<string> & Vec, const Table & T, const Table & TP, 
                                        RandDMVD_Index & RandDMVD_Index_subT, const long idx);
};
#endif