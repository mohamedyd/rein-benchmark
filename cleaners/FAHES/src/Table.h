/**************************************
 **** 2017-4-23      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#ifndef _DATA_TABLE_
#define _DATA_TABLE_

#include "common.h"


class Table {
public:
  long number_of_rows, number_of_cols;
  string table_name;
  vector<string> header;
  doubleVecStr data;
  
  Table() {}
  Table (const string &csv_file_name);
  Table (const string &name, 
          const long rows, 
          const long cols,
          vector<string> header, 
          doubleVecStr data
          );
  static vector<struct item> get_most_common(vector<map<string, long> > tabhist, long idx,
                                          long max_num_terms_per_att);
};


class TableProfile {
public:
  long number_of_tuples, number_of_atts;
  string table_name;
  vector<string> header;
  vector<profiler> profile;
  
  TableProfile() {}
  TableProfile (const string &csv_file_name);
  TableProfile (const string &name, 
          const long rows, 
          const long cols,
          vector<string> header, 
          vector<profiler> __profile
          );
  TableProfile (const Table & T);
};

#endif
