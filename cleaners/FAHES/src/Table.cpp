/**************************************
 **** 2017-4-23      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "Table.h"

// ============================================================================
// The definition of the member functions of Table Class
// ============================================================================

Table::Table(const string& name) {
  table_name = name;
}

Table::Table(const string &name,
             const long rows, 
             const long cols,
             vector<string> __header,
             doubleVecStr __data
            ){
  table_name = name;
  number_of_rows = rows;
  number_of_cols = cols;
  header = __header; 
  data = __data;
}


// ========= Find the most common values in an attribute ============
vector<struct item> Table::get_most_common(vector<map<string, long> > tabhist, long idx,
                                          long max_num_terms_per_att){
    vector<struct item> frequent;
    struct item freq_item;
    map<string, long> COL = tabhist[idx];
    long freq;
    long count = 0;
    long col_size = COL.size();
    
    while ((count < col_size) && ((long)frequent.size() < max_num_terms_per_att)){
        freq = -1;
        for(map<string, long>::iterator it = COL.begin(); it != COL.end(); it++){
            if (it->second > freq){
                string s = it->first;   trim(s);
                freq_item.value = s;
                freq_item.frequency = it->second;
                freq = it->second;
            }
        }
        count ++;
        if (freq_item.frequency == 1)
            break;
        COL.erase(freq_item.value);
        frequent.push_back(freq_item);
    }

    return frequent;
    
}

// ============================================================================
// The definition of the member functions of TableProfile Class
// ============================================================================
TableProfile::TableProfile (const string &name, 
          const long rows, 
          const long cols,
          vector<string> __header, 
          vector<profiler> __profile
          ){
  table_name = name;
  number_of_tuples = rows;
  number_of_atts = cols;
  header = __header; 
  profile = __profile;
}
// ============================================================================
TableProfile::TableProfile (const Table & T){
  table_name = T.table_name;
  number_of_tuples = T.number_of_rows;
  number_of_atts = T.number_of_cols;
  header = T.header;
}
