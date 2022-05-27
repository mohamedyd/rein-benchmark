/**************************************
 **** 2017-6-5      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "OD.h"
#include "density_estimator.h"

// ========================================================================

void OD::compute_statistical_quantities(map<double, long> & col_profile, double & mean, 
    									double & std){
	double sum = 0, sq_sum = 0;
	
	map<double, long>::iterator dbl_itr;
	long num_tuples = 0;
	dbl_itr = col_profile.begin();
	while (dbl_itr != col_profile.end()){
		sum += dbl_itr->first * dbl_itr->second;
		num_tuples += dbl_itr->second;
		sq_sum += pow(dbl_itr->first, EX) * dbl_itr->second;
		dbl_itr ++;
	}
	mean = sum / (double) num_tuples;
	std = compute_std(sum, sq_sum, num_tuples);
	// cerr << "Mean = " << mean << "\t STD = " << "\t Num. tuples = " << num_tuples << endl; 
}

// // ========================================================================
// vector<sus_disguised> OD::z_score_od(const string Attribute, map<double, long> & col_profile, 
//     									const double & mean, const double & std){
// 	vector<sus_disguised> sus_disg_vec;
// 	sus_disguised sus_disg;
// 	map<double, long>::iterator dbl_itr;
// 	double sc = 2.0, z_sc;
// 	string str;
// 	dbl_itr = col_profile.begin();
// 	while(dbl_itr != col_profile.end()){
// 		z_sc = z_score(dbl_itr->first, mean, std);
// 		if (z_sc > 3 * std){
// 			std::ostringstream strs;
// 			strs << dbl_itr->first;
// 			str = strs.str();
// 			sus_disg = prepare_sus_struct(Attribute, str, sc, dbl_itr->second);
// 			sus_disg_vec.push_back(sus_disg);
// 		}
// 		dbl_itr ++;
// 	}
// 	return sus_disg_vec;
// }
// ========================================================================
void OD::detect_outliers(TableProfile & TP, 
    			std::vector<sus_disguised> & sus_dis_values){
	
	map<double, long> numeric_data;
	map<double, long>::iterator dbl_itr;	//, dbl_ir2;
	vector<sus_disguised> sus_disg;
	vector<vector<sus_disguised> >	od_dvd(num_od_tools);
	double mean, std;
	long L_Str, L_Nums;
	for (long i = 0; i < (long)TP.header.size(); i++){
		L_Str = TP.profile[i].distinct_Strings.size();
		L_Nums = TP.profile[i].distinct_Numbers.size();
		// if (TP.header[i] == "Bldg Assignable Square Footage")
		// 	cerr << "Processing the att. (Bldg Assignable Square Footage)  " << L_Str << "\t" << L_Nums << endl;
		if (L_Str < num_non_ignorable_strings){
			dbl_itr = TP.profile[i].distinct_Numbers.begin();
			while (dbl_itr != TP.profile[i].distinct_Numbers.end()){
				numeric_data[double(dbl_itr->first)] = dbl_itr->second;
				dbl_itr ++;
			}		
			if ((long)numeric_data.size() >= least_num_values){
				compute_statistical_quantities(numeric_data, mean, std);
				// od_dvd[0] = z_score_od(TP.header[i], numeric_data, mean, std);
				// ================ For testing the density estimator =======================
				// string of_name = "Den/"+TP.header[i];
				// of_name += ".txt";
				// ofstream ofs(of_name, ios::out);
				// if (!ofs.good()){
				// 	cerr << "Unable to open output file to store the density.. \n";
				// 	continue;
				// }
				// Den_Estimator DE;
				// DE.build_PDF_model(numeric_data, std, min_val, max_val);
				// for (int kk = 0; kk < DE.den_fn.size(); kk ++){
				// 	ofs << DE.den_fn[kk].val << '\t' << DE.den_fn[kk].density << endl;
				// }
				// ofs.close();
				Den_Estimator DE;
				// cerr << "Attribute #( " << i << " ) has ( " << numeric_data.size() << " ) distinct values\n";
				sus_disg = DE.density_based_od(TP.header[i], numeric_data, TP.profile[i].common_Strings, std);
				// ================ For testing the density estimator =======================
				for (int k = 0; k < (long)sus_disg.size(); k++){
					if (!member_of(sus_disg[k], sus_dis_values))
						sus_dis_values.push_back(sus_disg[k]);
				}
				// cerr << TP.header[i] << "::Min = " << min_val << "::Max = " << max_val << endl;
			}
			numeric_data.clear();
			sus_disg.clear();
		}
		// ===========================================================
		// -- Hakim, this piece of code checks the frequency of the items to ...
		// find outliers in the frequency ...
		// (samples with very high frequency compared to the others)
		// ===========================================================
		// compute_statistical_quantities(TP.profile[i].freq_of_freq, mean, std);
		// Den_Estimator DE;
		// map <string, long> common;
		// map <double, long>::iterator ll_itr = TP.profile[i].freq_of_freq.begin();
		// for (;ll_itr != TP.profile[i].freq_of_freq.end(); ll_itr++){
		// 	string SSS = std::to_string(ll_itr->first);
		// 	common[SSS] = ll_itr->second;
		// }
		// sus_disg = DE.density_based_od(TP.header[i], TP.profile[i].freq_of_freq, common,
		// 			std, min_val, max_val);
		// // cerr << "Standard deviation = " << std << endl;
		// // ================ For testing the density estimator =======================
		// map <long, vector<string> >::iterator st_itr;
		// for (int k = 0; k < sus_disg.size(); k++){
		// 	long val = convert_to_long(sus_disg[k].value);
		// 	st_itr = TP.profile[i].sorted_Strings_by_freq.find(val);
		// 	for (int kk = 0; kk < st_itr->second.size(); kk++){
		// 		sus_disguised s_disguised = prepare_sus_struct(TP.header[i], st_itr->second[kk], 
		// 			1.0, val);
		// 		if (!member_of(s_disguised, sus_dis_values))
		// 			sus_dis_values.push_back(s_disguised);
		// 	}
		// }
	}
}



