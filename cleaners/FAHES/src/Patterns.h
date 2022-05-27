/**************************************
 **** 2017-10-29      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#ifndef _PATTERNS_H_
#define _PATTERNS_H_
#include "common.h"
#include "Profiler.h"
#include <limits.h>
		
#define LOWER 	'l'			// [a-z]
#define UPPER 	'u'			// [A-Z]
#define DIGIT 	'd'			// [0-9]
#define SPACE 	's'			
#define ALPHA 	'a'			// [A-Z]|[a-z]
#define ALNUM	'x'			// [A-Z]|[a-z]|[0-9]
#define DASH 	'h'
#define DOT 	't'				
#define COMMA 	'c'			// 
#define SYMBOL 	'y'			// Other unspecified symbols
#define PUNCT 	'p'			// Punctuation marks :;?/
#define ENCLOSE	'e'			// Enclosing symbols (),[] and {}
#define WORDP 	'w'			// Collection of alphabetic character ans few symbols {-,' .}
#define AND		'&'
#define HASH	'#'
#define AT 		'@'
#define PERCENT	'%'
#define POWER 	'^'
#define ASTRSK	'*'
#define NOT		'!'			// 
#define SQUOTE	'q'			// Single Quotation
#define USCR		'_'			// Underscore
#define SLASH	'/'
#define SPALPHA 'v'			// Special characters in few names such as {\"e}

class pattern_learner{
public:
	// vector<vector<string> > patterns;
	pattern_learner(){}
	// map<long, long>	patterns_hist;
	vector<string> L1_pattern(string str);
	void find_all_patterns(vector<map<string, long> > & tabhist, 
										TableProfile & TP,
										vector<sus_disguised> & sus_dis_values);
	void L1_patterns(map<string, long> & colhist, vector<vector<string> > & ptrn_vec,
									 vector<string> & ptrns_str,
									 map<string, long> & pttrns_hist);
	vector<string> L2_patterns(vector<vector<string> > & pttrns,
										map<string, long> & old_pttrns_hist);
	vector<string> L3_patterns(vector<vector<string> > & pttrns,
										map<string, long> & old_pttrns_hist);
	vector<string> L4_patterns(vector<vector<string> > & pttrns,
										map<string, long> & old_pttrns_hist);
	vector<string> L5_patterns(vector<vector<string> > & pttrns,
										map<string, long> & old_pttrns_hist);
	vector<string> aggregate_UL_classes(vector<string> ptrn);
	vector<string> aggregate_ASCHT_classes(vector<string> ptrn);
	vector<string> aggregate_WD_classes(vector<string> ptrn);
	vector<string> remove_numbers(vector<string> & old_ptrn);
	vector<string> get_cells_with_given_pattern(vector<string> ptrn, map<string, long> & colhist, long agglvl);
	void determine_dominating_patterns(map<string, long> pttrns_hist, map<string, bool> & dom_ptrns);
	string get_cell_pttrn(string val, long AGG_Level);
	vector<string> apply_tricks(vector<string> ptrn);
	vector<string> remove_enclosing(vector<string> ptrn);
	bool check_enclosing(vector<string> ptrn);
	vector<string> apply_L4_tricks(vector<string> ptrn);
	vector<string> apply_L5_tricks(vector<string> ptrn);

	// =================== Member function with definitions ==================
	bool Special_Alphabet(char ch){
		// cout << endl << (ushort)ch << endl;
		ushort cch = 255 - (USHRT_MAX - (ushort)ch);
		// cout << endl << cch << endl;
		if ((cch >= 1) && (cch <=8))			return true;
		if ((cch >= 11) && (cch <=26))			return true;
		if ((cch >= 192) && (cch <=214))		return true;
		if ((cch >= 217) && (cch <=246))		return true;
		if ((cch >= 249) && (cch <=254))		return true;
		return false;
	}
	// ==============================================================
	char get_char_class(char ch){
		if (isdigit(ch)) 			return DIGIT;
		if (islower(ch)) 			return LOWER;
		if (isupper(ch)) 			return UPPER;
		if (Special_Alphabet(ch))	return SPALPHA;
		if ((ch == ':') || (ch == '?')) 	return PUNCT;
		if ((ch == ' ') || (ch == '\t'))	return SPACE;
		if ((ch == '(') || (ch == '{') || (ch == '[')) return ENCLOSE;
		if ((ch == ')') || (ch == '}') || (ch == ']')) return ENCLOSE;
		if (ch == '-')	return DASH;
		if (ch == '.')	return DOT;
		if (ch == ',')	return COMMA;
		if (ch == '&')	return AND;
		if (ch == '#')	return HASH;
		if (ch == '@')	return AT;
		if (ch == '%')	return PERCENT;
		if (ch == '^')	return POWER;
		if (ch == '*')	return ASTRSK;
		if (ch == '!')	return NOT;
		if (ch == '\'')	return SQUOTE;
		if (ch == '_')	return USCR;
		if (ch == ';')	return PUNCT;
		if (ch == '/')	return SLASH;
		return SYMBOL;
	}
	// ==============================================================
	bool match_L3(char cch, char ch){
		if ((cch == ALPHA) && (ch == LOWER))		return true;
		if ((cch == ALPHA) && (ch == UPPER))		return true;
		if ((cch == ALPHA) && (ch == USCR))			return true;
		if (cch == ch)		return true;
		return false;
	}
	// ==============================================================
	bool match_L4(char cch, char ch){
		if ((cch == WORDP) && (ch == SQUOTE))	return true;
		if ((cch == WORDP) && (ch == SPACE))	return true;
		if ((cch == WORDP) && (ch == DASH))		return true;
		if ((cch == WORDP) && (ch == COMMA))	return true;
		if ((cch == WORDP) && (ch == DOT))		return true;
		if ((cch == WORDP) && (ch == ALPHA))	return true;
		if (cch == ch)							return true;
		return false;
	}
	// ==============================================================
	bool match_L5(char cch, char ch){
		if ((cch == WORDP) && (ch == DIGIT))	return true;
		if ((cch == WORDP) && (ch == SPALPHA))	return true;
		if (cch == ch)							return true;
		return false;
	}
	// ==============================================================
	// bool match_L6(char cch, char ch){
	// 	if ((ch == SPALPHA) && (cch == WORDP))	return true;
	// 	if (cch == ch)							return true;
	// 	return false;
	// }
};

#endif