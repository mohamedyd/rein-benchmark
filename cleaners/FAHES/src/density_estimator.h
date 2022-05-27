/**************************************
 **** 2017-6-3      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#ifndef _Den_Estimator_
#define _Den_Estimator_



#include "Table.h"
#include "common.h"

class Den_Estimator {
public:
	const int num_samples = 100;
	struct den_val{
		double val, density;
	};
	typedef std::vector<den_val> den_func;
	den_func den_fn;

    Den_Estimator(){ }
    // void print_den(den_func df);
    // void print_den(map<double, vector<double> >);
    double compute_bandwidth(double std, long n);
    // void lin_space(double min, double max);
    // double evaluate_pnt_by_interpolation(double x);
    double compute_max_pdf(map <double, long> & col_profiler, 
                double min_val, double max_val, double h);
    double evaluate_pnt(map <double, long> & col_profiler, const double & x,
					const double & h, const long & Size);
    // void build_PDF_model(map <double, long> & col_profiler, const double & h,
				// 	const double & min_val, const double & max_val, long S);
    vector<sus_disguised> density_based_od(const string Attribute, map<double, long> & col_profile, 
                                        map<string, long> & common, 
    									const double & std);
};

#endif 