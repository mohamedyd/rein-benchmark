/**************************************
 **** 2017-6-3      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "DV_Detector.h"
#include "OD.h"
#include "Profiler.h"
#include "RandDMVD.h"


void compute_statistical_quantities(const Table & T, vector<double> & subT_mean, 
								vector<double> & subT_std){
	double m, s, sum, sq_sum, val;
	long i, j;
	for (i = 0; i < T.number_of_cols; i++){
		sum = 0;	sq_sum = 0;
		for (j = 0; j < T.number_of_rows; j++) {
			val = convert_to_double(T.data[j][i]);
			sum += val;
			sq_sum += val * val;
		}
		s = compute_std(sum, sq_sum, T.number_of_rows);
		m = sum / (double) T.number_of_rows;
		subT_mean[i] = m;
		subT_std[i] = s;
	}
}

// // ========================================================================



// ========================================================================
void DV_Detector::detect_single_char_strings(TableProfile TP, 
    			std::vector<sus_disguised> & sus_dis_values){
	sus_disguised sus_dis;
	string s;
	map<string, long>::iterator string_itr;
	map<string, long>::iterator string_itr2;
	long L_Str, L_Str1, L_Nums;
	char c;
	double sc = 1.0;
	for (long i = 0; i < (long)TP.header.size(); i++){
		// cerr << TP.header[i] << endl;
		L_Str = TP.profile[i].distinct_Strings.size();
		L_Str1 = TP.profile[i].common_Strings.size();
		L_Nums = TP.profile[i].distinct_Numbers.size();
		// cout << L_Str << '\t' << L_Str1 << '\t' << L_Nums << endl;
		if ((L_Str <= 2) && (L_Str > 0)){
			if (L_Nums > 2){
				string_itr = TP.profile[i].distinct_Strings.begin();
				for (long j = 0; j < L_Str; j++){
					s = string_itr->first;
					string_itr2 = TP.profile[i].common_Strings.find(s);
					if (!(string_itr2 == TP.profile[i].common_Strings.end())){
						sus_dis = prepare_sus_struct(TP.header[i], s, sc, string_itr->second, "SYN");
						if (!member_of(sus_dis, sus_dis_values))
							sus_dis_values.push_back(sus_dis);
					}
					string_itr++;
				}
			}
		}
		string_itr = TP.profile[i].common_Strings.begin();
		for (long k = 0; k < L_Str1; k++){
			s = string_itr->first;
			if (s.length() == 1){
				c = s[0];
				if (!(isalnum(c))){
					string_itr2 = TP.profile[i].common_Strings.find(s);
					if (!(string_itr2 == TP.profile[i].common_Strings.end())){
						sus_dis = prepare_sus_struct(TP.header[i], s, sc, string_itr->second, "SYN");
						if (!member_of(sus_dis, sus_dis_values))
							sus_dis_values.push_back(sus_dis);
					}
				}
			}
		}
	}
}
// ========================================================================
void DV_Detector::positive_negative_inconsistency(TableProfile TP, 
    			std::vector<sus_disguised> & sus_dis_values){
	sus_disguised sus_dis;
	map<double, long>::iterator double_itr, double_itr2;
	long num_positives, num_negatives;
	map<double, long> positives, negatives;
	map<string, long>::iterator string_itr;
	long L_Nums;
	char c;
	double sc = 1.0;
	std::string str;
	long number_of_diff_ele = 2;
	for (long i = 0; i < (long)TP.header.size(); i++){
		num_positives = 0;	 num_negatives = 0;
		positives.clear();	 negatives.clear();
		double_itr = TP.profile[i].distinct_Numbers.begin();
		
		while (double_itr != TP.profile[i].distinct_Numbers.end()){
			if (double_itr->first >= 0){
				num_positives ++;
				positives[double_itr->first] = double_itr->second;
			}
			else{
				num_negatives ++;
				negatives[double_itr->first] = double_itr->second;
			}
			double_itr ++;
		}
		
		if ((num_positives == 1) && (num_negatives > number_of_diff_ele)){
			double_itr = positives.begin();
			std::ostringstream strs;
			strs << double_itr->first;
			str = strs.str();
			string_itr = TP.profile[i].common_Strings.find(str);
			if (!(string_itr == TP.profile[i].common_Strings.end())){
				sus_dis = prepare_sus_struct(TP.header[i], str, sc, double_itr->second, "SYN");
				if (!member_of(sus_dis, sus_dis_values))
					sus_dis_values.push_back(sus_dis);
			}
		}
		if ((num_positives > number_of_diff_ele) && (num_negatives == 1)){
			// cout << "Attribute = " << TP.header[i] << "\tNegatives = " << num_negatives 
			// 	 << "\tpositives = " << num_positives << endl;
			double_itr = negatives.begin();
			double_itr2 = positives.begin();
			std::ostringstream strs;
			strs << double_itr->first;
			str = strs.str();
			string_itr = TP.profile[i].common_Strings.find(str);
			if (!(string_itr == TP.profile[i].common_Strings.end())){
				sus_dis = prepare_sus_struct(TP.header[i], str, sc, double_itr->second, "SYN");
				if (!member_of(sus_dis, sus_dis_values))
					sus_dis_values.push_back(sus_dis);
			}
		}
	}
}
// ========================================================================
void DV_Detector::check_repeated_substrings(TableProfile TP,
				vector<map<string, long> > & M,  
    			std::vector<sus_disguised> & sus_dis_values){
	sus_disguised sus_disg;
	map<string, long>::iterator string_itr;
	double std_dev;
	double sc = 1.0;
	double threshold = 0.1;
	int num_rep_substr;			// The number of strings containing repeated substrings

	for (long i = 0; i < (long)TP.header.size(); i++){
		map<string, long>::iterator itr = M[i].begin();
		num_rep_substr = 0;
		while (itr != M[i].end()) {
			string s = itr->first;
			transform( s.begin(), s.end(), s.begin(), ::tolower );
			std_dev = check_str_repetition(s);
			// if (s == "00000000") cerr << "Got it1 (" << s << "," << std_dev << ")\n";
			if (std_dev == 0)
				num_rep_substr ++;
			itr++;
		}
		if (num_rep_substr > 0){
			// cout << TP.header[i] << '\t';
			// cout << "num_strs_containing_rep_substr = " << num_rep_substr << "\tOut of :" << M[i].size() << endl;
			if ((double) num_rep_substr < (threshold * (double) M[i].size())){			
				string_itr = TP.profile[i].common_Strings.begin();
				while (string_itr != TP.profile[i].common_Strings.end()){
					string s = string_itr->first;
					// if (TP.header[i] == "Hr Org Unit Id") cerr << "Common (" << s << ")\n";
					transform( s.begin(), s.end(), s.begin(), ::tolower );
					std_dev = check_str_repetition(s);
					// if (s == "00000000") cerr << "Got it2 (" << s << "," << std_dev << ")\n";
					if (std_dev == 0){
						sus_disg = prepare_sus_struct(TP.header[i], string_itr->first, sc, string_itr->second, "SYN");
						if (!member_of(sus_disg, sus_dis_values))
							sus_dis_values.push_back(sus_disg);
					}
					string_itr ++;
				}
			}
		}
	}
}
// ========================================================================
void DV_Detector::check_non_conforming_patterns(TableProfile & TP, 
				vector<map<string, long> > & M,
    			std::vector<sus_disguised> & sus_dis_values){
	// cerr << "Start : detect_single_char_strings\n"; 
	detect_single_char_strings(TP, sus_dis_values);
	// cerr << "Done : detect_single_char_strings\n"; 
	positive_negative_inconsistency(TP, sus_dis_values);
	// cerr << "Done : positive_negative_inconsistency\n";
	check_repeated_substrings(TP, M, sus_dis_values);
	// cerr << "Done : check_repeated_patterns\n";
}



