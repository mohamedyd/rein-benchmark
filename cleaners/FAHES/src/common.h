/**************************************
 **** 2017-4-23      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/
  
#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath> 
#include <iomanip>      // std::setprecision
#include <numeric>
#include <map>
#include <sstream>
#include <math.h>
#include <limits.h>

// #ifdef __APPLE__
// #include <unordered_map>
// #include <unordered_set>
// #else
// #include <tr1/unordered_map>
// #include <tr1/unordered_set>
// #endif

// 
// #ifndef __APPLE__
// using namespace tr1;
// #endif

using namespace std;
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

typedef vector<vector<string> > doubleVecStr;

struct profiler{
    long num_distinct_values, num_numerics, num_strings, num_nulls;
    map <string, long> distinct_Strings;
    map <double, long> distinct_Numbers;
    map <long, vector<string> > sorted_Strings_by_freq;
    map <string, long> common_Strings;
    map <double, long> freq_of_freq;
    void reset_profiler(){
      num_distinct_values = 0;
      num_numerics = 0;
      num_strings = 0;
      num_nulls = 0;
      distinct_Numbers.clear();
      distinct_Strings.clear();
      common_Strings.clear();
      sorted_Strings_by_freq.clear();
      freq_of_freq.clear();
    }
};

struct item{
    string value;
    long frequency;
};

struct sus_disguised{
  	string value, attr_name, tool_name;
    double score;
    long frequency;
};

bool equals(const vector<string> & V1, const vector<string> & V2, const long & idx);
double compute_KL(const double, const double, const double, const double);
double compute_std(const double & sum, const double & sq_sum, const long & n);
unsigned hash_str(const string word);
void remove_elements_in(vector<string> &V);
bool isNumber(const string& s);
bool isNULL(const string& s);
double convert_to_double(const string& s);
long convert_to_long(const string& s);
long check_data_type(const string& s );
bool check_num_repetition(const string);
double check_str_repetition(const string);
void trim(string& s);
void Print_double_vector(doubleVecStr V);
void print_line_of(char c);
void print_line_of(char c, std::fstream& fs);
string check_d_quotation(string str);
void print_vector(const vector<string> &V);
void print_vector(const vector<double> &V);
double kernel_func(double x);
bool member_of(const sus_disguised &, const vector<sus_disguised> &);
bool member_of(const string & , const vector<item> &);
bool member_of(const vector<string> & vec_ele, const doubleVecStr & com_vec);
void sort_sus_values(vector<sus_disguised> & );
sus_disguised prepare_sus_struct(string att, string s, double score, long freq, string tool_name);

#endif

