/**************************************
 **** 2017-6-6      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "density_estimator.h"


// ==============================================================
// void insert( den_func &cont, int value ) {
//     den_func::iterator it = std::lower_bound( cont.begin(), cont.end(), value, greater(den_val) ); // find proper position in descending order
//     cont.insert( it, value ); // insert before iterator it
// }
// ==============================================================
double Den_Estimator::compute_bandwidth(double std, long n){
	if (std == 0)	return 1.0;
	double bandwidth = (double)(2.345 * std * pow(n, -0.2));
	return bandwidth;
}
// // ==============================================================
// void Den_Estimator::lin_space(double min, double max){
// 	long n = num_samples;
// 	den_val d_val;
// 	int count = 0;
// 	double eval_step = (max - min) / (double)(n - 1);
// 	for (double e_pnt = min; e_pnt < max; e_pnt += eval_step){
// 		d_val.val = e_pnt;
// 		d_val.density = 0.0;
// 		den_fn.push_back(d_val);
// 		count ++;
// 	}
// 	cout << count;
// 	d_val.val = max;
// 	d_val.density = 0.0;
// 	den_fn.push_back(d_val);
// }

// ==============================================================
double Den_Estimator::compute_max_pdf(map <double, long> & col_profile, 
				double min_val, double max_val, double h){
	long n = 100;
	double eval_step = (max_val - min_val) / (double)(n);
	double max_pdf = 0, pdf;
	long S = col_profile.size();
	for (double e_pnt = min_val; e_pnt < max_val; e_pnt += eval_step){
		pdf = evaluate_pnt(col_profile, e_pnt, h, S);
		if (max_pdf < pdf)
			max_pdf = pdf;
	}
	return max_pdf;
}


// // ==============================================================

// void Den_Estimator::build_PDF_model(map <double, long> & col_profiler, const double & h,
// 					const double & min_val, const double & max_val, long S){
// 	map <double, long>::iterator it;
	
// 	double x;
// 	while(!den_fn.empty()){
//         den_fn.pop_back();
//     }
// 	lin_space(min_val - (2 * h), max_val + (2 * h));
// 	for (int i = 0; i < den_fn.size(); i++){
// 		x = den_fn[i].val;
// 		den_fn[i].density = evaluate_pnt(col_profiler, x, h, S);
// 	}
// }

// ==============================================================

double Den_Estimator::evaluate_pnt(map <double, long> & col_profiler, const double & x,
					const double & h, const long & S){
	double epdf; 
	map <double, long>::iterator itr;
	double sum = 0.0;
	for (itr = col_profiler.begin(); itr != col_profiler.end(); itr++){
		sum += kernel_func((x - itr->first) / h) / h * itr->second;
	}
	epdf = (sum / ((double)S));
	// cout << x << '\t' << epdf << endl;
	return epdf;
}
// ==============================================================
// double Den_Estimator::evaluate_pnt_by_interpolation(double x){
// 	double epdf, eval_step, w1, w2;
// 	eval_step = den_fn[2].val - den_fn[1].val;
// 	int idx = (x - den_fn[1].val) / eval_step;
// 	if ((idx < 0) || (idx >= den_fn.size()))		return 0.0;
// 	if ((den_fn[idx].val <= x) && (den_fn[idx+1].val > x)){
// 		w1 = x - den_fn[idx].val;
// 		w2 = den_fn[idx+1].val - x;
// 		epdf = ((w1 * den_fn[idx+1].density) + (w1 * den_fn[idx+1].density)) / (w1 + w2);
// 		return epdf;
// 	}
// 	cerr << "Interpolation Error " << endl;
// 	return 0.0;
// }

// ==============================================================
vector<sus_disguised> Den_Estimator::density_based_od(const string Attribute, map<double, long> & col_profile,
										map<string, long> & common, 
    									const double & std){
	vector<sus_disguised> sus_disg;
	int num_distinct_vals = col_profile.size();
	map<string, long>::iterator str_itr;
	map<double, long>::iterator dbl_itr1, temp_dbl_itr, tt_dbl_itr;
	den_func::iterator dbl_itr2;

	// den_func df;
	const int ensembles = 4;
	den_val dv;
	sus_disguised sus_val;
	// map<double, vector<double> > density_func;
	// map<double, vector<double> >::iterator den_itr;
	long S = 0, outlier_freq;
	double epdf, val, max_score = 0, f_i, min_val, max_val, f_max;
	string outlier;
	double min_dist = std, score;
	temp_dbl_itr = col_profile.begin();
	tt_dbl_itr = col_profile.end();
	tt_dbl_itr--;
	min_val = temp_dbl_itr->first;
	max_val = tt_dbl_itr->first;
	for (dbl_itr1 = col_profile.begin(); dbl_itr1 != col_profile.end(); dbl_itr1++){
		S += dbl_itr1->second;
		temp_dbl_itr ++;
		if (temp_dbl_itr != col_profile.end())
			min_dist = MIN(min_dist, std::abs(temp_dbl_itr->first - dbl_itr1->first));
	}	
	double h0 = compute_bandwidth(std, S);
	f_max = compute_max_pdf(col_profile, min_val, max_val, h0);
	if ((min_dist <= h0) && (col_profile.size() > 10)){
		for (str_itr = common.begin(); str_itr != common.end(); str_itr++){
			epdf = f_max;
			if (!isNumber(str_itr->first))		continue;
			for (int kk = 0; kk < ensembles; kk++){
				val = convert_to_double(str_itr->first);
				double h = h0 - (0.2 * kk *h0);
				f_i = evaluate_pnt(col_profile, val, h, S);
				if (epdf > f_i)
					epdf = f_i;
			}
			score = MAX(f_max - epdf, 0.0) / f_max;
			// dv.val = val;
			// dv.density = epdf;
			// df.push_back(dv);
			if (score > max_score){
				max_score = score;
				outlier = str_itr->first;
				outlier_freq = str_itr->second;
			}

			// den_itr = density_func.find(epdf);
			// density_func[epdf].push_back(dv.val);
			// if (epdf <= 1E-16){
			// 	sus_val = prepare_sus_struct(Attribute, str_itr->first, 1, str_itr->second, "OD");
			// 	if(!member_of(sus_val, sus_disg))
			// 		sus_disg.push_back(sus_val);
			// }        
		}
		if (max_score > (1.0 - 1E-16)){
			sus_val = prepare_sus_struct(Attribute, outlier, max_score, outlier_freq, "OD");
			if(!member_of(sus_val, sus_disg))
				sus_disg.push_back(sus_val);
		}   


		// int limit = MIN(3, df.size());
		// // for (int i = 0; i < limit; i ++){
		// // 	// cout << Attribute << "::" << dbl_itr2->second << "::" << col_profile[dbl_itr2->second] << endl;
		// // 	cout << Attribute << "::" << df[i].val << endl;
		// // }
		// int count = 0;
		// for (den_itr = density_func.begin(); den_itr != density_func.end(); den_itr ++){
		// 	for (int i = 0; i < den_itr->second.size(); i++){
		// 		cout << Attribute << "::" << den_itr->second[i] << "::" << den_itr->first << endl;
		// 		count ++;
		// 		if (count >= limit)
		// 			break;
		// 	}
		// 	if (count >= limit)
		// 		break;
		// }
		// cout << "===================================================\n";
	}
	// if (Attribute == "Diastolic blood pressure")
	// 	print_den(density_func);
	return sus_disg;
}

// // ==============================================================
// void Den_Estimator::print_den(den_func df){
// 	ofstream ofs("Den.txt", ios::out);
// 	if (!ofs.good()){
// 		cerr << "Unable to open the output file to print the density values \n";
// 		return;
// 	}
// 	for (int i = 0; i < df.size(); i++)
// 		ofs << df[i].val << "\t" << df[i].density << endl;
// 	ofs.close();
// }

// ==============================================================
// void Den_Estimator::print_den(map<double, vector<double> > density_func){
// 	map<double, vector<double> >::iterator den_itr;
// 	ofstream ofs("Den.txt", ios::out);
// 	if (!ofs.good()){
// 		cerr << "Unable to open the output file to print the density values \n";
// 		return;
// 	}
// 	for (den_itr = density_func.begin(); den_itr != density_func.end(); den_itr ++)
// 		for (int i = 0; i < den_itr->second.size(); i++)
// 			ofs << den_itr->second[i] << '\t' << den_itr->first << endl;
// 	ofs.close();
// }