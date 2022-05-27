/**************************************
 **** 2017-3-6      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "Profiler.h"
#include "common.h"


// ========================================================================



// ========================================================================
// Measure the inhirited information from the original database 
// in the projected database
// ========================================================================

float DataProfiler::compare_distribution(const long N1, const long N2, 
                            const long idx,
                            vector<map<string, long> > & M1, 
                            vector<map<string, long> > & M2){
    map<string, long> Q = M1[idx];
    map<string, long> P = M2[idx];
    map<string, long>::iterator itr = P.begin();

    float KL = 0, p, q, A;
    float freq1, freq2;
    while (itr != P.end()) {
            freq2 = itr->second;
            freq1 = Q.find(itr->first)->second;
            p = float(freq2) / float(N2);
            q = float(freq1) / float(N1);
            // KL += p * log(p/q);
            KL += fabs(p-q);
            itr++;
        }
    return KL;
}
// ========================================================================
// bool DataProfiler::prune_value(const string & S, const vector<map<string, long> > & Part_M, 
//                             const vector<map<string, long> > & M){
//     return true;
// }
// ========================================================================
bool DataProfiler::prune_attribute(const long idx, const long len, const vector<map<string, long> > & M){
    if ((long)M[idx].size() < 3)
        return true;
    if ((long)M[idx].size() == len)
        return true;
    return false;
}
// ========================================================================
long DataProfiler::find_least_distinct_values(vector<map<string, long> > tabhist)
{
    long idx = 0, num_distinct_values = tabhist[0].size();
    for (long i = 1; i < (long)tabhist.size(); i++){
        if((long)tabhist[i].size() < num_distinct_values){
            idx = i;
            num_distinct_values = tabhist[i].size();
        }
    }
    return idx;
}


// ============ Find the distinct values in each column and their frequency =================
vector<map<string, long> > DataProfiler::TableHistogram(const Table & table)
{
        vector<map<string, long> > m_tablehist;
        long num_col = table.number_of_cols;
        string SS;
        m_tablehist = vector<map<string, long> > (num_col);
        for (long i = 0; i < (long)table.data.size(); i++) {
        for(long j = 0; j < (long)table.data[0].size(); j++) {
            SS = table.data[i][j];
            transform( SS.begin(), SS.end(), SS.begin(), ::tolower );
            if ((SS.empty()) || (SS == "null") || (SS.length() == 0)){
                map<string, long>::iterator itr = m_tablehist[j].find("NULL");
                if (itr == m_tablehist[j].end()) {
                        m_tablehist[j]["NULL"] = 1;
                } else {
                        itr->second++;
                }
            }
            else{
                map<string, long>::iterator itr = m_tablehist[j].find(table.data[i][j]);
                if (itr == m_tablehist[j].end()) {
                        m_tablehist[j][table.data[i][j]] = 1;
                } else {
                        itr->second++;
                }
            }
        }
    }
        return m_tablehist;
}

// ============ Print the distinct values in each column and their frequency =================
void DataProfiler::PrintTableHist(vector<map<string, long> > m_tablehist) 
{
	for (long i = 0; i < (long)m_tablehist.size(); i++) {
		cout << "***************" << endl;
		cout << "column: " << i << " histogram" << endl;
		map<string, long>::iterator itr = m_tablehist[i].begin();
		while (itr != m_tablehist[i].end()) {
			cout << itr->first << ": " << itr->second << endl;
			itr++;
		}
	}
}
// ============ Print the number of values of each data type =================
TableProfile DataProfiler::profile_table(const Table & T, vector<map<string, long> > & TabHist){
    string s;
    long freq;
    TableProfile TP(T);
    long type;
    vector<struct item> most_frequent;
    map <long, vector<string> >::iterator string_vec_itr;
        
    for (long i = 0; i < (long)TabHist.size(); i++){
        map<string, long>::iterator itr = TabHist[i].begin();
        profiler PR;
        PR.reset_profiler();
        PR.num_distinct_values = TabHist[i].size();
        while (itr != TabHist[i].end()) {
            s = itr->first;
            freq = itr->second;
            trim(s);
            type = check_data_type(s);
            switch(type)
            {
                case 1:{    PR.num_numerics += itr->second;
                            double num_value = convert_to_double(s);
                            PR.distinct_Numbers[num_value] = itr->second;
                            PR.sorted_Strings_by_freq[itr->second].push_back(s);
                            break;
                        }

                case 2:{    PR.num_nulls += itr->second;           break;}

                case 3:{    PR.num_strings += itr->second;
                            PR.distinct_Strings[s] = itr->second;
                            PR.sorted_Strings_by_freq[itr->second].push_back(s);
                            break;
                        }
            }
            itr++;
            map<double, long>::iterator dbl_itr = PR.freq_of_freq.find(freq);
            if (!(s == "NULL")){
                if (dbl_itr == PR.freq_of_freq.end()) {
                    PR.freq_of_freq[freq] = 1;
                } else {
                    dbl_itr->second++;
                }
            }
        }
        TP.profile.push_back(PR);
    }
    for (long i = 0; i < (long)TabHist.size(); i++){
        if (TP.profile[i].sorted_Strings_by_freq.size() > 1){
            long count = 0;
            bool done = 0;
            string_vec_itr = TP.profile[i].sorted_Strings_by_freq.end();
            // cerr << "Values = " << TP.profile[i].sorted_Strings_by_freq.size() << endl;
            string_vec_itr --;
            while (string_vec_itr != TP.profile[i].sorted_Strings_by_freq.begin()){
                    for (size_t k = 0; k < string_vec_itr->second.size(); k++){
                        if (string_vec_itr->first > 1){
                            TP.profile[i].common_Strings[string_vec_itr->second[k]] = string_vec_itr->first;
                            count ++;
                        }
                        else { done = 1;    break;}        
                    }
                if (done)   break;
                string_vec_itr --;
            }
        }
    }
    return TP;
}




