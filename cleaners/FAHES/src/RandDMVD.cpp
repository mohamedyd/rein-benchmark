/**************************************
 **** 2017-7-25      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "RandDMVD.h"
#include "Profiler.h"

// ========================================================================
RandDMVD::RandDMVD(){
    for (long i = 0; i < (long)RandDMVD_Index_T.size(); i++){
        RandDMVD_Index_T[i].clear();        
    }
}

// ========================================================================
long RandDMVD::compute_num_cooccur(const vector<string> & Vec, const long & idx){
    long num_coocur = 0;
    long index_min_subT;
    long min_subT;
    bool first_time = true;
    for (long i = 0; i < (long)Vec.size(); i++){ 
        if (i != idx){
            map<string, doubleVecStr>::iterator index_itr11 = RandDMVD_Index_T[i].find(Vec[i]);
            if (first_time){
                min_subT = index_itr11->second.size();
                index_min_subT = i;
                first_time = false;
            }
            else{
                if (min_subT > (long)index_itr11->second.size()){
                    min_subT = index_itr11->second.size();
                    index_min_subT = i;
                }
            }
        }
    }
    map<string, doubleVecStr>::iterator index_itr = RandDMVD_Index_T[index_min_subT].find(Vec[index_min_subT]);
    for (long K = 0; K < (long)index_itr->second.size(); K++){
        if (equals(index_itr->second[K], Vec, idx))     num_coocur++;
    }
    return num_coocur;
}

// ========================================================================
long RandDMVD::compute_num_occur(string v, long A){
    map<string, doubleVecStr>::iterator index_itr12 = RandDMVD_Index_T[A].find(v);
    return index_itr12->second.size();
}

// ========================================================================
long RandDMVD::compute_num_cooccur(const vector<string> & Vec, const long & idx,
                                    RandDMVD_Index & RandDMVD_Index_subT){
    long num_coocur = 0;
    long index_min_subT;
    long min_subT;
    bool first_time = true;
    for (long i = 0; i < (long)Vec.size(); i++){ 
        if (i != idx){
            map<string, doubleVecStr>::iterator index_itr11 = RandDMVD_Index_subT[i].find(Vec[i]);
            if (first_time){
                min_subT = index_itr11->second.size();
                index_min_subT = i;
                first_time = false;
            }
            else{
                if (min_subT > (long)index_itr11->second.size()){
                    min_subT = index_itr11->second.size();
                    index_min_subT = i;
                }
            }
        }
    }
    map<string, doubleVecStr>::iterator index_itr = RandDMVD_Index_T[index_min_subT].find(Vec[index_min_subT]);
    for (long K = 0; K < (long)index_itr->second.size(); K++){
        if (equals(index_itr->second[K], Vec, idx))     num_coocur++;
    }
    return num_coocur;
}

// ========================================================================
long RandDMVD::compute_num_occur(string v, long A, RandDMVD_Index & RandDMVD_Index_subT){
    map<string, doubleVecStr>::iterator index_itr12 = RandDMVD_Index_subT[A].find(v);
    return index_itr12->second.size();
}
// ========================================================================
bool RandDMVD::prune_attribute(const long idx, vector<map<string, long> > & M){
    if (M[idx].size() < 3)
        return true;
    // if ((long)M[idx].size() == len)
    //     return true;
    return false;
}
// ========================================================================
double RandDMVD::subtable_correlation(const Table & T, const string mc,
                                        const long idx, long & num_rows_in_subtable){
    double corr = 0;
    map<string, doubleVecStr>::iterator index_itr = RandDMVD_Index_T[idx].find(mc);
    Table TP; 
    TP.header = T.header;
    TP.number_of_cols = T.number_of_cols;
    TP.number_of_rows = index_itr->second.size();
    TP.data = index_itr->second;
    RandDMVD_Index RandDMVD_Index_subT;
    Table_Index_RandDMVD(TP, RandDMVD_Index_subT);
    num_rows_in_subtable = index_itr->second.size();
    for (long i = 0; i < TP.number_of_rows; i++){
        corr +=  record_correlation(TP.data[i], T, TP, RandDMVD_Index_subT, idx);
    }
    return corr;
}

// ========================================================================
// double RandDMVD::Attribute_correlation(const Table & T, const Table & TP, RandDMVD_Index & RandDMVD_Index_subT,
//                                         const long idx){
    
//     double corr_T, corr_PT, corr = 0;
//     long num_rows_in_subtable = TP.number_of_rows;
//     // vector<string> Vec(2);
//     // doubleVecStr VecVec;
//     // cerr << index_itr->first << "::" << num_rows_in_subtable;
//     for (i = 0; i < num_rows_in_subtable; i++){
//                 corr +=  record_correlation(TP[i], T, TP, RandDMVD_Index_subT, idx);
//             // }
//         }
//     return corr;
// }
// ============== SELECT subtable when attribute equals a given value =========
// Table RandDMVD::SELECT(const Table & T, const string & v, const long & A){
//     Table TT;
//     TT.table_name = T.table_name;
//     TT.header = T.header;
//     map<string, doubleVecStr>::iterator index_itr = RandDMVD_Index_T[A].find(v);
//     TT.data = index_itr->second;
//     TT.number_of_rows = index_itr->second.size();
//     TT.number_of_cols = index_itr->second[0].size();
//     return TT;
// }
// ========================================================================
double RandDMVD::record_correlation(const vector<string> & Vec, const Table & T, const Table & TP, 
                                        RandDMVD_Index & RandDMVD_Index_subT, const long idx){
    long n = T.number_of_cols;
    double corr = 0, corr_t, corr_pt;
    double p_t, p_pt, p_V_t, p_V_pt;
    vector<double> p_vi_t(n), p_vi_pt(n);
    vector<long> n_vi_t(n), n_vi_pt(n);
    double q = 1;       // -- Order of Minkowski distance
    long i, j, n_V_t, n_V_pt;
    long num_rows_in_subtable = TP.number_of_rows;
    
    // cerr << "(" << v1 << ',' << v2 << "), ";
    // cerr << "(" << att1 << ',' << att2 << ")\n";
    // cerr << "::" << index_itr->first << "::" << num_rows_in_subtable << endl;
    for (long i = 0; i < n; i ++){
        string s = Vec[i];
        transform( s.begin(), s.end(), s.begin(), ::tolower );
        if ((s.empty()) || (s == "null") || (s.length() == 0))   return 0.0;
    }
    // for (i = 0; i < T.number_of_rows; i ++){
    //     if (T.data[i][att1] == v1){
    //         n_vi_t ++;
    //         if (T.data[i][att2] == v2)  n_vi_vj_t ++;
    //     }
    //     if (T.data[i][att2] == v2)  n_vj_t ++;
    // }
    n_V_t = compute_num_cooccur(Vec, idx);
    for (long i = 0; i < n; i ++)
        if (i != idx)   n_vi_t[i] = compute_num_occur(Vec[i], i);
    
    n_V_pt = compute_num_cooccur(Vec, idx, RandDMVD_Index_subT);
    for (long i = 0; i < n; i ++)
        if (i != idx)   n_vi_pt[i] = compute_num_occur(Vec[i], i, RandDMVD_Index_subT);

    // for (i = 0; i < num_rows_in_subtable; i ++){
    //     if (index_itr->second[i][att1] == v1){
    //         n_vi_pt ++;
    //         if (index_itr->second[i][att2] == v2)  n_vi_vj_pt ++;
    //     }
    //     if (index_itr->second[i][att2] == v2)  n_vj_pt ++;
    // }
    p_V_t = (double)n_V_t / (double) T.number_of_rows;
    p_V_pt = (double)n_V_pt / (double) num_rows_in_subtable;
    p_t = 1;
    p_pt = 1;
    for (long i = 0; i < n; i ++){
        if (i != idx){
            p_vi_t[i] = (double)n_vi_t[i] / (double) T.number_of_rows;
            p_t *= p_vi_t[i];
            p_vi_pt[i] = (double)n_vi_pt[i] / (double) num_rows_in_subtable;
            p_pt *= p_vi_pt[i];
        } 
    }
    
    corr_t = p_V_t / p_t;
    corr_pt = p_V_pt / p_pt;
    corr = p_V_t / (1 + pow(abs(corr_t - corr_pt), q));
    return corr;    
}

// ========================================================================
void RandDMVD::Table_Index_RandDMVD(const Table & T, RandDMVD_Index & IDX){
    // RandDMVD_Index IDX;
    vector<map<string, doubleVecStr> > m_tablehist;
    
    vector<string> row;
    long num_col = T.number_of_cols;
        string SS;
        m_tablehist = vector<map<string, doubleVecStr> > (num_col);
        row = vector<string> (num_col);
        for (long i = 0; i < T.number_of_rows; i++) {
            for(long j = 0; j < T.number_of_cols; j++) {
                for (long ii = 0; ii < T.number_of_cols; ii++)
                    row[ii] = T.data[i][ii];
                SS = T.data[i][j];
                transform( SS.begin(), SS.end(), SS.begin(), ::tolower );
                if ((SS.empty()) || (SS == "null") || (SS.length() == 0)){
                    map<string, doubleVecStr>::iterator itr = m_tablehist[j].find("NULL");
                    if (itr == m_tablehist[j].end()) {
                        doubleVecStr T_data;
                        T_data.push_back(row);
                        m_tablehist[j]["NULL"] = T_data;
                    } else {
                            itr->second.push_back(row);
                    }
                }
                else{
                    map<string, doubleVecStr>::iterator itr = m_tablehist[j].find(T.data[i][j]);
                    if (itr == m_tablehist[j].end()) {
                        doubleVecStr T_data;
                        T_data.push_back(row);
                        m_tablehist[j][T.data[i][j]] = T_data;
                    } else {
                            itr->second.push_back(row);
                    }
                }
        }
    }
    IDX = m_tablehist;
    // return IDX;
}

// ========================================================================
vector< sus_disguised > RandDMVD::find_disguised_values(const Table & T,
                                    vector<map<string, long> > & tablehist,
                                    long max_num_terms_per_att){
    std::vector<sus_disguised> sus_dis_values, sus_dis_values_per_att;
    long PT_num_rows;
    Table Temp_T;
    DataProfiler DP;
    vector<map<string, long> > Temp_tablehist;
    Temp_T.number_of_rows = T.number_of_rows;
    vector<long> KK;
    for (long i = 0; i < T.number_of_cols; i++){
        if (!prune_attribute(i, tablehist)){
            Temp_T.header.push_back(T.header[i]);
            KK.push_back(i);
        }
    }
    Temp_T.number_of_cols = KK.size();
    if ((int) KK.size() == 1)
        return sus_dis_values;
    // cout << "The size of the new table = ( " << Temp_T.number_of_rows << " , " << Temp_T.number_of_cols << " )\n";
    std::vector<string> row(Temp_T.number_of_cols);
    for (long i = 0; i < T.number_of_rows; i++){
        for (long j = 0; j < (long)KK.size(); j++)
            row[j] = T.data[i][KK[j]];
        Temp_T.data.push_back(row);
    }
    Table_Index_RandDMVD(Temp_T, RandDMVD_Index_T);
    Temp_tablehist = DP.TableHistogram(Temp_T);
    sus_disguised dis_value;
    vector<struct item> most_com;
    double a, b, c, d, ratio = 1.3, corr, DV_Score;
    // long i = 7;
    for (long i = 0; i < Temp_T.number_of_cols; i++){
        // cout << "=====================================\n";
        // cout << T.header[i] << endl;        
        most_com = Table::get_most_common(Temp_tablehist, i, max_num_terms_per_att);
        for(long k = 0; k < (long)most_com.size(); k++){                
            // PT = SELECT(T, most_com[k].value, i);
            // corr = subtable_correlation(T, PT, i);
            string str = most_com[k].value;
            transform( str.begin(), str.end(), str.begin(), ::tolower );
            if (str == "null")        continue;
            corr = subtable_correlation(Temp_T, most_com[k].value, i, PT_num_rows);
            DV_Score = (double)Temp_T.number_of_rows / (double) PT_num_rows * corr;
            dis_value.attr_name = Temp_T.header[i];
            dis_value.value = most_com[k].value;
            dis_value.score = DV_Score;
            dis_value.frequency = most_com[k].frequency;
            dis_value.tool_name = "Rand";
            sus_dis_values_per_att.push_back(dis_value);
            // cout << corr << ',' << DV_Score << endl;
            // cout << most_com[k].value << '\t' << most_com[k].frequency << endl;
        }
        sort_sus_values(sus_dis_values_per_att);
        // for (long L = 0; L < 5; L ++){
        if (sus_dis_values_per_att.size() > 0){
            dis_value = sus_dis_values_per_att[0];
            double ratio1 = (double)dis_value.frequency / (double)Temp_T.number_of_rows;
            double ratio2 = (double)Temp_tablehist[i].size() / (double)Temp_T.number_of_rows;
            if ((ratio1 > 0.01) && (ratio2 > 0.01) && (dis_value.frequency > 5))
                sus_dis_values.push_back(dis_value);
        }
        // }
        while(!sus_dis_values_per_att.empty())
            sus_dis_values_per_att.pop_back();
        // cout << "=====================================\n"; 
    }
    return sus_dis_values;
}
