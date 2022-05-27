/**************************************
 **** 2017-4-23      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#include "common.h"
#include <stdlib.h>


bool equals(const vector<string> & V1, const vector<string> & V2, const long & idx){
    bool res = true;
    if (V1.size() != V2.size())  return false;
    for (long i = 0; i < (long)V1.size(); i++){
        if (i == idx)   continue;
        if (V1[i] != V2[i])     return false;
    }
    return true;
}
// ========================================================================
void sort_sus_values(vector<sus_disguised> & sus_dis_values_per_att){
    sus_disguised temp;
    long i, j;
    for (i = 0; i < (long)sus_dis_values_per_att.size(); i ++)
        for (j = i + 1; j < (long)sus_dis_values_per_att.size(); j++){
            if (sus_dis_values_per_att[i].score < sus_dis_values_per_att[j].score){
                temp = sus_dis_values_per_att[j];
                sus_dis_values_per_att[j] = sus_dis_values_per_att[i];
                sus_dis_values_per_att[i] = temp;
            }
        }
} 
// ========================================================================
double compute_KL(const double mu1, const double mu2, const double s1, const double s2){
    float power = 2.0;
    double mu_d = pow((mu1-mu2), power) / (2*pow(s1, power));
    double s_d = 0.5 * (pow((s2/s1), power) - 1.0 - log(pow(s2 / s1, power)));
    double KL = mu_d + s_d;
    return KL;
}
// ========================================================================
double compute_std(const double & sum, const double & sq_sum, const long & n){
        double sigma = pow(((sq_sum - (1.0 / (double) n) * sum * sum))/(double)(n - 1), 0.5);
        return sigma;
    }
// ========================================================================
unsigned hash_str(const string word){
    #define A 54059 /* a prime */
    #define B 76963 /* another prime */
    #define C 86969 /* yet another prime */
    #define FIRSTH 37 /* also prime */

    unsigned h = FIRSTH;
    for(long i = 0; i < (long)word.length(); i++) {
      h = (h * A) ^ (word[i] * B);
    }
    return h; // or return h % C;
}
// ========================================================================
string check_d_quotation(string str){
    std::size_t found = str.find(",");
    if ((found!=std::string::npos) || (str.empty()))
        return "\""+str+"\"";
    return str;
}


// ========================================================================
void remove_elements_in(vector<string> &V){
    while(!V.empty()){
        V.pop_back();
    }
}

// ========================================================================
void print_vector(const vector<string> &V){
//    cout << "[ " ;
    for (long i = 0; i < (long)V.size() - 1; i++)
         cout << check_d_quotation(V[i]) << ',';
    cout << check_d_quotation(V[V.size()-1]) << "\n";
}

// ========================================================================
void print_vector(const vector<double> &V){
//    cout << "[ " ;
    for (long i = 0; i < (long)V.size() - 1; i++)
         cout << V[i] << '\t';
    cout << V[V.size()-1] << "\n";
}

// ========================================================================
void print_double_vector(doubleVecStr V){
    for (long i = 0; i < (long)V.size(); i++)
        print_vector(V[i]);
}

// ========================================================================
void print_line_of(char c){
    for (long i = 0; i < 30; i++)
        cout << c;
    cout << endl;
}
void print_line_of(char c, std::fstream& fs){
    for (long i = 0; i < 30; i++)
        fs << c;
    fs << endl;
}



// ========================================================================
void trim(string& s)
{
    size_t p = s.find_first_not_of(" \t");
    s.erase(0, p);
    
    p = s.find_last_not_of(" \t");
    if (string::npos != p)
        s.erase(p+1);
}

// ========================================================================
bool isNumber(const string& s) {
    string str = s;
    str.erase(std::remove(str.begin(), str.end(), ','), str.end());
    if ((str[0]=='%') || (str[str.length()-1]=='%')){
        size_t n = std::count(str.begin(), str.end(), '%');

        if (n == 1)
            str.erase(std::remove(str.begin(), str.end(), '%'), str.end());
    }
    std::istringstream iss(str);
    double f;
    iss >> noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail(); 
}

// ========================================================================

double convert_to_double(const string& s){
    string str = s;
    str.erase(std::remove(str.begin(), str.end(), ','), str.end());
    if ((str[0]=='%') || (str[str.length()-1]=='%')){
        size_t n = std::count(str.begin(), str.end(), '%');
        if (n == 1)
            str.erase(std::remove(str.begin(), str.end(), '%'), str.end());
    }
    std::istringstream iss(str);
    // double f = std::stod(str);
    double f;
    iss >> noskipws >> f;
    return f;
}

// ========================================================================

long convert_to_long(const string& s){
    string str = s;
    str.erase(std::remove(str.begin(), str.end(), ','), str.end());
    if ((str[0]=='%') || (str[str.length()-1]=='%')){
        size_t n = std::count(str.begin(), str.end(), '%');
        if (n == 1)
            str.erase(std::remove(str.begin(), str.end(), '%'), str.end());
    }
    std::istringstream iss(str);
    // double f = std::stod(str);
    long L;
    iss >> noskipws >> L;
    return L;
}

// ========================================================================

bool isNULL(const string& s){
    string SS = s;
    transform( SS.begin(), SS.end(), SS.begin(), ::tolower );
    if (s.empty() || (SS == "null"))    return true;
    return false;
}
// ========================================================================
long check_data_type(const string& s ){
    long type;
    if (isNumber(s))     return 1;
    if (isNULL(s))      return 2;
    return 3;
}
// ========================================================================
// double check_str_repetition(const string str){
//     int min_dist_idx, min_dist;
//     double s = 0, sqs = 0, count = 0, std_dev;
//     if (str.length() < 5)
//         return 1.0;
//     min_dist = abs(str[0] - str[1]);
//     min_dist_idx = 1;
//     for (int j = 1; j < (int)str.length(); j++){
//         if (abs(str[0] - str[j]) < min_dist){
//             min_dist = abs(str[0] - str[j]);
//             min_dist_idx = j;
//         }
//     }
//     // if ((!isNumber(str)) && (min_dist_idx > 1))
//     if (min_dist_idx > 1)
//         return 1.0;

//     if (min_dist_idx > (int)(str.length()/2))
//         for (int k = 0; k < (int)str.length() - 1; k++){
//             s += abs(str[k] - str[k+1]);
//             sqs += abs(str[k] - str[k+1]) * abs(str[k] - str[k+1]);
//             count ++;
//         }
//     else
//         for (int k = 0; k < (int)str.length() - min_dist_idx; k++){
//             double a = abs(str[k] - str[k+min_dist_idx]);
//             s += a;
//             sqs += a * a;
//             count ++;
//         }
//     std_dev = compute_std(s, sqs, count);
    
//     return std_dev;
// }
// ========================================================================
double check_str_repetition(const string str){
    int min_dist_idx, min_dist;
    double s = 0, sqs = 0, count = 0, std_dev;
    if (str.length() < 5)
        return 1.0;
    min_dist = abs(str[0] - str[1]);
    min_dist_idx = 1;
    for (int j = 1; j < (int)str.length(); j++){
        if (abs(str[0] - str[j]) < min_dist){
            min_dist = abs(str[0] - str[j]);
            min_dist_idx = j;
        }
    }
    // if ((!isNumber(str)) && (min_dist_idx > 1))
    if (min_dist_idx > 1)
        return 1.0;

    if (min_dist_idx > (int)(str.length()/2))
        for (int k = 0; k < (int)str.length() - 1; k++){
            s += str[k] - str[k+1];
            sqs += (str[k] - str[k+1]) * (str[k] - str[k+1]);
            count ++;
        }
    else
        for (int k = 0; k < (int)str.length() - min_dist_idx; k++){
            double a = str[k] - str[k+min_dist_idx];
            s += a;
            sqs += a * a;
            count ++;
        }
    std_dev = compute_std(s, sqs, count);
    
    return std_dev;
}
// // ========================================================================
// double check_str_repetition(const string val){
//     int rep_threshold = 5;
//     long longest_rep_str = 0;
//     if ((long)val.length() < rep_threshold)
//         return 1.0;
//     for (long i = 0; i < val.length() - 1; i++){
//         if (!isdigit(val[i])){
//             if (val[i] != val[i+1]) 
//                 longest_rep_str = 0;
//             else{
//                 if (longest_rep_str == 0)
//                     longest_rep_str = 2;
//                 else
//                     longest_rep_str ++;
//             }
//             if(longest_rep_str >= rep_threshold)
//                 return 0.0;
//         }
//     }
//     return 1.0;
// }
// ========================================================================
sus_disguised prepare_sus_struct(string att, string s, double score, long freq, string tool){
  sus_disguised sus_dis_struct;
  sus_dis_struct.attr_name = att;
  sus_dis_struct.value = s;
  sus_dis_struct.score = score;
  sus_dis_struct.frequency = freq;
  sus_dis_struct.tool_name = tool;
  return sus_dis_struct; 
}

// ========================================================================
double kernel_func(double x)
{
    if((fabs(x) <= 1) && (fabs(x) != 0))
        return (0.75 * (1 - x * x));
    else
        return (0.0);
}
// ========================================================================
bool member_of(const sus_disguised & s_ele, const vector<sus_disguised> & s_vec){
    for (long i = 0; i < (long)s_vec.size(); i ++)
        if ((s_ele.value == s_vec[i].value) && (s_ele.attr_name == s_vec[i].attr_name))
            return true;
    return false;
}

// ========================================================================
bool member_of(const string & s_ele, const vector<item> & com_vec){
    for (long i = 0; i < (long)com_vec.size(); i ++)
        if (s_ele == com_vec[i].value)
            return true;
    return false;
}


// ========================================================================
bool member_of(const vector<string> & vec_ele, const doubleVecStr & com_vec){
    bool found;
    for (long i = 0; i < (long)com_vec.size(); i ++){
        found = true;
        for (long j = 0; j < (long)vec_ele.size(); j++)
            if (vec_ele[j] != com_vec[i][j]){
                found = false;
                break;
            }
        if (found)  return true;
    }
    return false;
}


// ===============================EOF======================================
