/**************************************
 **** 2017-10-29      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/
#include "Patterns.h"

// ==============================================================
long convert_string_to_int(string str){
	std::istringstream iss(str);
    // double f = std::stod(str);
    long L;
    iss >> noskipws >> L;
    return L;
}
// ==============================================================
void check_and_push(vector<string> & results_ptrn){
	string W_plus = "w+";
	if (results_ptrn.empty())
		{ 	results_ptrn.push_back(W_plus); 		return;		}
	string ptrn = results_ptrn[results_ptrn.size()-1];
	if (ptrn[0] == 'w'){
			results_ptrn[results_ptrn.size()-1] = W_plus;
			// cout << "check1" << endl;
	}
	else {
		results_ptrn.push_back(W_plus);
		// cout << "check2";
	}
}
// ==============================================================
vector<string> pattern_learner::apply_L4_tricks(vector<string> ptrn){
	vector<string> results_ptrn, temp_ptrn;
	while(!results_ptrn.empty())	results_ptrn.pop_back();
	
	if (ptrn.size() >= 3){
		int i = 0;
		
		while (i < (int) ptrn.size()-2){
			if ((ptrn[i][0] == HASH) && ((ptrn[i+1][0] == SPACE)) && ((ptrn[i+2][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == HASH) && ((ptrn[i+1][0] == SPACE)) && ((ptrn[i+2][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			results_ptrn.push_back(ptrn[i]);	i ++;
		}
		for (int k = i; k < (int)ptrn.size(); k++)
			results_ptrn.push_back(ptrn[k]);
		ptrn = results_ptrn;
	}
	while(!results_ptrn.empty())	results_ptrn.pop_back();
	if (ptrn.size() >= 2){
		int i = 0;
		while (i < (int) ptrn.size()-1){
			if ((ptrn[i][0] == HASH) && ((ptrn[i+1][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 2; 		continue;	}
			if ((ptrn[i][0] == HASH) && ((ptrn[i+1][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 2; 		continue;	}
			results_ptrn.push_back(ptrn[i]);	i ++;
		}
		for (int k = i; k < (int)ptrn.size(); k++)
			results_ptrn.push_back(ptrn[k]);
	}
	if(!results_ptrn.empty()){
		// ptrn_str = "";
		// for (int k = 0; k < (int) results_ptrn.size(); k++)
		// 	ptrn_str += results_ptrn[k];
		// cout << ptrn_str << endl;
		return results_ptrn;
	}
	return ptrn;
}
// ==============================================================
vector<string> pattern_learner::apply_L5_tricks(vector<string> ptrn){
	vector<string> results_ptrn, temp_ptrn;
	while(!results_ptrn.empty())	results_ptrn.pop_back();
	
	string ptrn_str = "";
	for (int k = 0; k < (int) ptrn.size(); k++)
		ptrn_str += ptrn[k];

	// cout << "L4 Tricks is applied on : " << ptrn_str << '\t';

	if (ptrn.size() == 3){
		if ((ptrn[0][0] == DIGIT) && ((ptrn[1][0] == DOT)) && ((ptrn[1].length() == 1)) && ((ptrn[2][0] == DIGIT)))
			results_ptrn.push_back("d+");
	}
	if (ptrn.size() >= 3){
		int i = 0;		
		while (i < (int) ptrn.size()-2){
			if ((ptrn[i][0] == DIGIT) && ((ptrn[i+1][0] == DASH)) && ((ptrn[i+2][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == WORDP) && ((ptrn[i+1][0] == AT)) && ((ptrn[i+2][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == WORDP) && ((ptrn[i+1][0] == AND)) && ((ptrn[i+2][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == SPACE) && ((ptrn[i+1][0] == AND)) && ((ptrn[i+2][0] == SPACE)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == SPACE) && ((ptrn[i+1][0] == AND)) && ((ptrn[i+2][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == WORDP) && ((ptrn[i+1][0] == AND)) && ((ptrn[i+2][0] == SPACE)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == WORDP) && ((ptrn[i+1][0] == SLASH)) && ((ptrn[i+2][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == DIGIT) && ((ptrn[i+1][0] == SLASH)) && ((ptrn[i+2][0] == WORDP)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == WORDP) && ((ptrn[i+1][0] == SLASH)) && ((ptrn[i+2][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == DIGIT) && ((ptrn[i+1][0] == SLASH)) && ((ptrn[i+2][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			if ((ptrn[i][0] == HASH) && ((ptrn[i+1][0] == SPACE)) && ((ptrn[i+2][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 3; 		continue;	}
			results_ptrn.push_back(ptrn[i]);	i ++;
		}
		for (int k = i; k < (int)ptrn.size(); k++)
			results_ptrn.push_back(ptrn[k]);
		ptrn = results_ptrn;
	}
	while(!results_ptrn.empty())	results_ptrn.pop_back();
	if (ptrn.size() >= 2){
		int i = 0;		
		while (i < (int) ptrn.size()-1){
			if ((ptrn[i][0] == HASH) && ((ptrn[i+1][0] == DIGIT)))
				{	check_and_push(results_ptrn);		i+= 2; 		continue;	}
			results_ptrn.push_back(ptrn[i]);	i ++;
		}
		for (int k = i; k < (int)ptrn.size(); k++)
			results_ptrn.push_back(ptrn[k]);
		ptrn = results_ptrn;
	}
	if(!results_ptrn.empty()){
		return results_ptrn;
	}
	return ptrn;
}
// ==============================================================
vector<string> pattern_learner::remove_enclosing(vector<string> ptrn){
	vector<string> new_ptrn;
	while(!new_ptrn.empty())	new_ptrn.pop_back();
	if (check_enclosing(ptrn)){
		for (int i = 0; i < (int) ptrn.size(); i++)
			if (ptrn[i][0] != ENCLOSE)
				new_ptrn.push_back(ptrn[i]);
		ptrn = new_ptrn;
	}
	return ptrn;
}

// ==============================================================
bool pattern_learner::check_enclosing(vector<string> ptrn){
	int count = 0;
	vector<char> enc_stack;
	while (!enc_stack.empty())	enc_stack.pop_back();
	for (int i = 0; i < (int) ptrn.size(); i++){
		if (ptrn[i][0] == ENCLOSE){
			if (enc_stack.empty()) {	enc_stack.push_back(ENCLOSE);	}
			else {	enc_stack.pop_back();	}
		}
	}
	if (enc_stack.empty())
		return true;
	return false;
}
// ==============================================================
void pattern_learner::find_all_patterns(vector<map<string, long> > & tabhist, 
										TableProfile & TP,
										vector<sus_disguised> & sus_dis_values){
	map<string, long> col_hist;
	vector<vector<string> > ptrns_vec;
	map<string, long> pttrns_hist;
	map<string, long>::iterator ptrn_hist_itr;
	vector<string> ptrns_str;
	string test_ptrn, val;
	sus_disguised sus_dis;
	map<string, long> AGG_Level;	// The aggregation level of the classes applied on each attribute
	map<string, long>::iterator agg_itr;
	map<string, long>::iterator string_itr;
	for (long i = 0; i < (long)tabhist.size(); i++) {
		col_hist = tabhist[i];
		// cout << "========================================================================\n";
		// cout << "Current Attribute = " << TP.header[i] << '\t';
		L1_patterns(col_hist, ptrns_vec, ptrns_str, pttrns_hist);
		AGG_Level[TP.header[i]] = 1;
		agg_itr = AGG_Level.find(TP.header[i]);
		size_t min_num_ptrns = 5;
		if (pttrns_hist.size() > min_num_ptrns){
			L2_patterns(ptrns_vec, pttrns_hist);			
			agg_itr->second++;
		}
		if (pttrns_hist.size() > min_num_ptrns){
			L3_patterns(ptrns_vec, pttrns_hist);
			agg_itr->second++;
		}
		if (pttrns_hist.size() > min_num_ptrns){
			L4_patterns(ptrns_vec, pttrns_hist);
			agg_itr->second++;
		}
		if (pttrns_hist.size() > min_num_ptrns){
			L5_patterns(ptrns_vec, pttrns_hist);
			agg_itr->second++;
		}
		// Determine dominating patterns
		
		map<string, bool> dominating_pttrns;
		dominating_pttrns.clear();
		map<string, bool>::iterator dom_pttrns_itr;
		// cout << "========================================\n";
		// cout << TP.table_name << "::" << TP.header[i] << endl;
		determine_dominating_patterns(pttrns_hist, dominating_pttrns);
		// dom_pttrns_itr = dominating_pttrns.begin();
		// while(dom_pttrns_itr != dominating_pttrns.end()){
		// 	cout << dom_pttrns_itr->first << "\t{" << dom_pttrns_itr->second << "}\n";
		// 	dom_pttrns_itr++;
		// }
		string_itr = TP.profile[i].common_Strings.begin();
		while (string_itr != TP.profile[i].common_Strings.end()){
			string s = string_itr->first;
			test_ptrn = get_cell_pttrn(s, agg_itr->second);
			// for (dom_pttrns_itr = dominating_pttrns.begin(); dom_pttrns_itr != dominating_pttrns.end(); dom_pttrns_itr++){
			// 	if (test_ptrn == dom_pttrns_itr->first){
					dom_pttrns_itr = dominating_pttrns.find(test_ptrn);
					if(dom_pttrns_itr == dominating_pttrns.end()){
						cerr << "Pattern not found .. ";
					}
					else
					if (!dom_pttrns_itr->second){
						sus_dis = prepare_sus_struct(TP.header[i], s, 1.0, string_itr->second, "SYN");
						if (!member_of(sus_dis, sus_dis_values))
						sus_dis_values.push_back(sus_dis);
					}

			// 	}
			// }
			
			string_itr ++;
		}
		
		// for (ptrn_hist_itr = pttrns_hist.begin(); ptrn_hist_itr != pttrns_hist.end(); ptrn_hist_itr++){
		// 	cout << "{" << ptrn_hist_itr->first << "} {" << ptrn_hist_itr->second << "}\t";
		// 	vector<string> cells_with_pattern;
		// 	// if ((ptrn_hist_itr->second < 20) && pttrns_hist.size() < 20){
		// 	if (ptrn_hist_itr->second < 20){	
		// 		string test;
		// 		for (int k = 0; k < (int)ptrns_vec.size(); k++){
		// 			while (!cells_with_pattern.empty())		cells_with_pattern.pop_back();
		// 			test = "";
		// 			for (int j = 0; j < (int)ptrns_vec[k].size(); j++)
		// 				test += ptrns_vec[k][j];
		// 			if (test == ptrn_hist_itr->first){ 				
		// 				cells_with_pattern = get_cells_with_given_pattern(ptrns_vec[k], col_hist, agg_itr->second);
		// 				cout << "{";
		// 				for (int m = 0; m < (int) cells_with_pattern.size()-1; m++)
		// 					cout << check_d_quotation(cells_with_pattern[m]) << ", ";
		// 				cout << check_d_quotation(cells_with_pattern[cells_with_pattern.size()-1]) << "}";
		// 			}
		// 		}
		// 	}
		// 	cout << endl;
		// }
	}
}
// ==============================================================
void pattern_learner::L1_patterns(map<string, long> & colhist,
													vector<vector<string> > & ptrn_vec,
													vector<string> & ptrns_str,
													map<string, long> & pttrns_hist){	
	map<string, long>::iterator itr;
	vector<string> pttrns, pttrn;
	string ptrn;
	map<string, long>::iterator ptrn_hist_itr;
	// ========== Clear the data containers if they have elements from previous calls ========
	pttrns_hist.clear();
	while (!pttrns.empty())			pttrns.pop_back();
	while (!ptrns_str.empty())		ptrns_str.pop_back();
	while (!ptrn_vec.empty())		ptrn_vec.pop_back();

	for (itr = colhist.begin(); itr != colhist.end(); itr++){
		if (itr->first == "NULL") 	continue;
		pttrn = L1_pattern(itr->first);
		ptrn = "";
		for (int i = 0; i < (int) pttrn.size(); i++)
			ptrn += pttrn[i];
		// cout << "Cur pattern = " << ptrn << endl;
		ptrn_hist_itr = pttrns_hist.find(ptrn);
		if (ptrn_hist_itr == pttrns_hist.end()) {
			ptrn_vec.push_back(pttrn);
			pttrns_hist[ptrn] = 1;
		} else {
			ptrn_hist_itr->second++;
		}
	}
	// cout << "number of patterns = " << ptrn_vec.size() << endl;
	// cout << "number of unique patterns = " << pttrns_hist.size() << endl;
	for (ptrn_hist_itr = pttrns_hist.begin(); ptrn_hist_itr != pttrns_hist.end(); ptrn_hist_itr++){
		// cout << "{" << ptrn_hist_itr->first << "} {" << ptrn_hist_itr->second << "}\n";
		ptrns_str.push_back(ptrn_hist_itr->first);
	}
}
// ==============================================================
vector<string> pattern_learner::L1_pattern(string str){
	vector<string> pattern;
	char ch;
	while (!pattern.empty())	pattern.pop_back();
	for(int i = 0; i < (int)str.length(); i++){
		ch = get_char_class(str[i]);
		// cout << ch << endl;
		if (!pattern.empty()){
			string cur_class = pattern[pattern.size()-1];
			if (cur_class[0] == ch){
				// cout << cur_class << endl;
				if (ch == 's')	continue;
				if(cur_class.length() > 1){
					string num = cur_class.substr (1,cur_class.length());
					// cout << "Num = " << num << endl;
					long rep = convert_string_to_int(num);
					rep++;
					string ch_class(1,ch);
					pattern[pattern.size()-1] = ch_class + std::to_string(rep);
				}
				else{
					string ch_class(1,ch);
					pattern[pattern.size()-1] = ch_class + "2";
				}
			}
			else{
				string ch_class(1,ch);
				pattern.push_back(ch_class);
			}
		}
		else{
				string ch_class(1,ch);
				pattern.push_back(ch_class);
			}
	}
	// for (int k = 0; k < (int)pattern.size(); k++)
	// 	cout << pattern[k];
	// cout << endl;
	pattern = remove_enclosing(pattern);
	return pattern;
}
// ==============================================================
vector<string> pattern_learner::remove_numbers(vector<string> & old_ptrn){
	vector<string> temp_ptrn;
	while (!temp_ptrn.empty())		temp_ptrn.pop_back();
	for (long i = 0; i < (long)old_ptrn.size(); i++){
		string ch_class(1,old_ptrn[i][0]);
		if (old_ptrn[i].length() > 1)
			temp_ptrn.push_back(ch_class+ "+");
		else
			temp_ptrn.push_back(ch_class);
	}
	return temp_ptrn;
}
// ==============================================================
vector<string> pattern_learner::L2_patterns(vector<vector<string> > & pttrns,
													map<string, long> & old_pttrns_hist){
	vector<string> reduced_pttrns_str;
	vector<string> new_ptrn, old_ptrn;
	string new_ptrn_str, old_ptrn_str;
	map<string, long> new_pttrns_hist;
	map<string, long>::iterator new_ptrn_hist_itr; 
	map<string, long>::iterator old_ptrn_hist_itr;
	vector<vector<string> > reduced_pttrns_vec;
	while (!reduced_pttrns_vec.empty())		reduced_pttrns_vec.pop_back();
	for (long i = 0; i < (long) pttrns.size(); i++){
		while(!new_ptrn.empty())	new_ptrn.pop_back();
		new_ptrn_str = "";
		old_ptrn_str = "";
		old_ptrn = pttrns[i];
		new_ptrn = remove_numbers(old_ptrn);
		for (int k = 0; k < (int)old_ptrn.size(); k++)
			old_ptrn_str += old_ptrn[k];
		for (int k = 0; k < (int)new_ptrn.size(); k++)
			new_ptrn_str += new_ptrn[k];
		new_ptrn_hist_itr = new_pttrns_hist.find(new_ptrn_str);
		old_ptrn_hist_itr = old_pttrns_hist.find(old_ptrn_str);
		if (new_ptrn_hist_itr == new_pttrns_hist.end()) {
			new_pttrns_hist[new_ptrn_str] = old_ptrn_hist_itr->second;
			reduced_pttrns_vec.push_back(new_ptrn);
		} else {
			new_ptrn_hist_itr->second += old_ptrn_hist_itr->second;
		}
	}
	old_pttrns_hist = new_pttrns_hist;
	pttrns = reduced_pttrns_vec;
	return reduced_pttrns_str;
}
// ==============================================================
vector<string> pattern_learner::L3_patterns(vector<vector<string> > & pttrns,
													map<string, long> & old_pttrns_hist){
	vector<string> reduced_pttrns_str;
	vector<string> new_ptrn, old_ptrn;
	string new_ptrn_str, old_ptrn_str;
	map<string, long> new_pttrns_hist;
	map<string, long>::iterator new_ptrn_hist_itr; 
	map<string, long>::iterator old_ptrn_hist_itr;
	vector<vector<string> > reduced_pttrns_vec;
	while (!reduced_pttrns_vec.empty())		reduced_pttrns_vec.pop_back();
	for (long i = 0; i < (long) pttrns.size(); i++){
		while(!new_ptrn.empty())	new_ptrn.pop_back();
		new_ptrn_str = "";
		old_ptrn_str = "";
		old_ptrn = pttrns[i];
		new_ptrn = aggregate_UL_classes(old_ptrn);
		for (int k = 0; k < (int)old_ptrn.size(); k++)
			old_ptrn_str += old_ptrn[k];
		for (int k = 0; k < (int)new_ptrn.size(); k++)
			new_ptrn_str += new_ptrn[k];
		new_ptrn_hist_itr = new_pttrns_hist.find(new_ptrn_str);
		old_ptrn_hist_itr = old_pttrns_hist.find(old_ptrn_str);
		if (new_ptrn_hist_itr == new_pttrns_hist.end()) {
			new_pttrns_hist[new_ptrn_str] = old_ptrn_hist_itr->second;
			reduced_pttrns_vec.push_back(new_ptrn);
		} else {
			new_ptrn_hist_itr->second += old_ptrn_hist_itr->second;
		}
	}
	old_pttrns_hist = new_pttrns_hist;
	pttrns = reduced_pttrns_vec;
	return reduced_pttrns_str;
}

// ==============================================================
vector<string> pattern_learner::aggregate_UL_classes(vector<string> ptrn){
	vector<string> new_ptrn;
	char ch;
	while (!new_ptrn.empty())	new_ptrn.pop_back();
	for(int i = 0; i < (int)ptrn.size(); i++){
		ch = ptrn[i][0];
		if (match_L3(ALPHA, ch))	ch = ALPHA;
		if (!new_ptrn.empty()){
			char cur_class = new_ptrn[new_ptrn.size()-1][0];
			if (match_L3(cur_class, ch)){
				string ch_class(1,ch);
				new_ptrn[new_ptrn.size()-1] = ch_class + "+";
			}
			else{
				string ch_class(1,ch);
				if (ptrn[i].length() > 1)
					new_ptrn.push_back(ch_class + "+");
				else
					new_ptrn.push_back(ch_class);
			}
		}
		else{
				string ch_class(1,ch);
				if (ptrn[i].length() > 1)
					new_ptrn.push_back(ch_class + "+");
				else
					new_ptrn.push_back(ch_class);
			}
	}
	return new_ptrn;
}

// ==============================================================
vector<string> pattern_learner::L4_patterns(vector<vector<string> > & pttrns,
													map<string, long> & old_pttrns_hist){
	vector<string> reduced_pttrns_str;
	vector<string> new_ptrn, old_ptrn;
	string new_ptrn_str, old_ptrn_str;
	map<string, long> new_pttrns_hist;
	map<string, long>::iterator new_ptrn_hist_itr; 
	map<string, long>::iterator old_ptrn_hist_itr;
	vector<vector<string> > reduced_pttrns_vec;

	while (!reduced_pttrns_vec.empty())		reduced_pttrns_vec.pop_back();
	for (long i = 0; i < (long) pttrns.size(); i++){
		while(!new_ptrn.empty())	new_ptrn.pop_back();
		new_ptrn_str = "";
		old_ptrn_str = "";
		old_ptrn = pttrns[i];
		new_ptrn = aggregate_ASCHT_classes(old_ptrn);
		for (int k = 0; k < (int)old_ptrn.size(); k++)
			old_ptrn_str += old_ptrn[k];
		for (int k = 0; k < (int)new_ptrn.size(); k++)
			new_ptrn_str += new_ptrn[k];
		new_ptrn_hist_itr = new_pttrns_hist.find(new_ptrn_str);
		old_ptrn_hist_itr = old_pttrns_hist.find(old_ptrn_str);
		if (new_ptrn_hist_itr == new_pttrns_hist.end()) {
			new_pttrns_hist[new_ptrn_str] = old_ptrn_hist_itr->second;
			reduced_pttrns_vec.push_back(new_ptrn);
		} else {
			new_ptrn_hist_itr->second += old_ptrn_hist_itr->second;
		}
	}
	old_pttrns_hist = new_pttrns_hist;
	pttrns = reduced_pttrns_vec;
	return reduced_pttrns_str;
}
// ==============================================================
vector<string> pattern_learner::aggregate_ASCHT_classes(vector<string> ptrn){
	vector<string> new_ptrn;
	char ch;
	while (!new_ptrn.empty())	new_ptrn.pop_back();
	for(int i = 0; i < (int)ptrn.size(); i++){
		ch = ptrn[i][0];
		if(match_L4 (WORDP, ch))	ch = WORDP;
		if (!new_ptrn.empty()){
			char cur_class = new_ptrn[new_ptrn.size()-1][0];
			if(match_L4 (cur_class, ch)){
				string ch_class(1,ch);
				new_ptrn[new_ptrn.size()-1] = ch_class + "+";
			}
			else{
				string ch_class(1,ch);
				if (ptrn[i].length() > 1)
					new_ptrn.push_back(ch_class + "+");
				else
					new_ptrn.push_back(ch_class);
			}
		}
		else{
				string ch_class(1,ch);
				if (ptrn[i].length() > 1)
					new_ptrn.push_back(ch_class + "+");
				else
					new_ptrn.push_back(ch_class);
			}
	}
	return new_ptrn;
}


// ==============================================================
vector<string> pattern_learner::L5_patterns(vector<vector<string> > & pttrns,
													map<string, long> & old_pttrns_hist){
	vector<string> reduced_pttrns_str;
	vector<string> new_ptrn, old_ptrn;
	string new_ptrn_str, old_ptrn_str;
	map<string, long> new_pttrns_hist;
	map<string, long>::iterator new_ptrn_hist_itr; 
	map<string, long>::iterator old_ptrn_hist_itr;
	vector<vector<string> > reduced_pttrns_vec;

	while (!reduced_pttrns_vec.empty())		reduced_pttrns_vec.pop_back();
	for (long i = 0; i < (long) pttrns.size(); i++){
		while(!new_ptrn.empty())	new_ptrn.pop_back();
		new_ptrn_str = "";
		old_ptrn_str = "";
		old_ptrn = pttrns[i];
		new_ptrn = aggregate_WD_classes(old_ptrn);
		// new_ptrn = aggregate_WD_classes(old_ptrn);
		for (int k = 0; k < (int)old_ptrn.size(); k++)
			old_ptrn_str += old_ptrn[k];
		for (int k = 0; k < (int)new_ptrn.size(); k++)
			new_ptrn_str += new_ptrn[k];
		new_ptrn_hist_itr = new_pttrns_hist.find(new_ptrn_str);
		old_ptrn_hist_itr = old_pttrns_hist.find(old_ptrn_str);
		if (new_ptrn_hist_itr == new_pttrns_hist.end()) {
			new_pttrns_hist[new_ptrn_str] = old_ptrn_hist_itr->second;
			reduced_pttrns_vec.push_back(new_ptrn);
		} else {
			new_ptrn_hist_itr->second += old_ptrn_hist_itr->second;
		}
	}
	old_pttrns_hist = new_pttrns_hist;
	pttrns = reduced_pttrns_vec;
	return reduced_pttrns_str;
}
// ==============================================================
vector<string> pattern_learner::aggregate_WD_classes(vector<string> ptrn){
	vector<string> new_ptrn;
	char ch;
	while (!new_ptrn.empty())	new_ptrn.pop_back();
	ptrn = apply_L4_tricks(ptrn);
	for(int i = 0; i < (int)ptrn.size(); i++){
		ch = ptrn[i][0];
		if (match_L5 (WORDP, ch))	ch = WORDP;
		if (!new_ptrn.empty()){
			char cur_class = new_ptrn[new_ptrn.size()-1][0];
			if(match_L5 (cur_class, ch)){
				// ch = WORDP;				
				string ch_class(1,ch);
				new_ptrn[new_ptrn.size()-1] = "w+";
			}
			else{
				string ch_class(1,ch);
				if (ptrn[i].length() > 1)
					new_ptrn.push_back(ch_class + "+");
				else
					new_ptrn.push_back(ch_class);
			}
		}
		else{
				string ch_class(1,ch);
				if (ptrn[i].length() > 1)
					new_ptrn.push_back(ch_class + "+");
				else
					new_ptrn.push_back(ch_class);
			}
	}
	new_ptrn = aggregate_ASCHT_classes(new_ptrn);
	string old_ptrn_str, new_ptrn_str;
	vector<string> ptrn_after_tricks;
	do {
		ptrn_after_tricks = apply_L5_tricks(new_ptrn);
		old_ptrn_str = "";		new_ptrn_str = "";
		for (int k = 0; k < (int)new_ptrn.size(); k++)
			old_ptrn_str += new_ptrn[k];
		for (int k = 0; k < (int)ptrn_after_tricks.size(); k++)
			new_ptrn_str += ptrn_after_tricks[k];
		new_ptrn = ptrn_after_tricks;
		// cout << new_ptrn_str << '\t' << old_ptrn_str << endl;
	} while(new_ptrn_str != old_ptrn_str);
	return new_ptrn;
}



// ==============================================================
vector<string> pattern_learner::get_cells_with_given_pattern(vector<string> ptrn, 
											map<string, long> & colhist, long agglvl){
	map<string, long>::iterator itr = colhist.begin();
	string ptrn_str = "", new_ptrn_str;
	vector<string> ptrn_vec;
	vector<string> cells_with_X_pattern;
	while (!cells_with_X_pattern.empty())	cells_with_X_pattern.pop_back();
	for (int i = 0; i < (int)ptrn.size(); i++)
		ptrn_str += ptrn[i];

	while(itr != colhist.end()){
		ptrn_vec = L1_pattern(itr->first);
		switch (agglvl){
			case 1: break;
			case 2:
				ptrn_vec = remove_numbers(ptrn_vec);	break;
			case 3:
				ptrn_vec = remove_numbers(ptrn_vec);
				ptrn_vec = aggregate_UL_classes(ptrn_vec);
				break;
			case 4:
				ptrn_vec = remove_numbers(ptrn_vec);
				ptrn_vec = aggregate_UL_classes(ptrn_vec);
				ptrn_vec = aggregate_ASCHT_classes(ptrn_vec);
				break;
			case 5:
				ptrn_vec = remove_numbers(ptrn_vec);
				ptrn_vec = aggregate_UL_classes(ptrn_vec);
				ptrn_vec = aggregate_ASCHT_classes(ptrn_vec);
				ptrn_vec = aggregate_WD_classes(ptrn_vec);
				break;
			default: cout << "unknown aggregation level .. \n";
		}
		new_ptrn_str = "";
		for (int i = 0; i < (int)ptrn_vec.size(); i++)
			new_ptrn_str += ptrn_vec[i];
		if (new_ptrn_str == ptrn_str){
			// cout << "{" << itr->first << "}\t has \t{" << new_ptrn_str 
			// 	 << "} Pattern\n";
			cells_with_X_pattern.push_back(itr->first);
		}
		itr ++;
	}
	return cells_with_X_pattern;
}



// ==============================================================
string pattern_learner::get_cell_pttrn(string val, long agglvl){
	vector<string> ptrn_vec;
	while (!ptrn_vec.empty())		ptrn_vec.pop_back();
	ptrn_vec = L1_pattern(val);
	switch (agglvl){
		case 1: break;
		case 2:
			ptrn_vec = remove_numbers(ptrn_vec);	break;
		case 3:
			ptrn_vec = remove_numbers(ptrn_vec);
			ptrn_vec = aggregate_UL_classes(ptrn_vec);
			break;
		case 4:
			ptrn_vec = remove_numbers(ptrn_vec);
			ptrn_vec = aggregate_UL_classes(ptrn_vec);
			ptrn_vec = aggregate_ASCHT_classes(ptrn_vec);
			break;
		case 5:
			ptrn_vec = remove_numbers(ptrn_vec);
			ptrn_vec = aggregate_UL_classes(ptrn_vec);
			ptrn_vec = aggregate_ASCHT_classes(ptrn_vec);
			ptrn_vec = aggregate_WD_classes(ptrn_vec);
			break;
		default: cout << "unknown aggregation level .. \n";
	}
	string new_ptrn_str = "";
	for (size_t i = 0; i < ptrn_vec.size(); i++)
		new_ptrn_str += ptrn_vec[i];
	return new_ptrn_str;
}


// ==============================================================
void pattern_learner::determine_dominating_patterns(map<string, long> pttrns_hist, 
													map<string, bool> & dom_ptrns){
	map<long, vector<string> > ptrns_by_freq;
	map<long, vector<string> >::reverse_iterator freq_itr;
	map<long, vector<string> >::iterator f_itr;
	map<string, long>::iterator itr = pttrns_hist.begin();
	long total_dist_vals = 0;
	
	while(itr != pttrns_hist.end()){
		// f_itr = ptrns_by_freq.find(itr->second);
		total_dist_vals += itr->second;
		// if (freq_itr != ptrns_by_freq.end()){
		// 	ptrns_by_freq->second.push_back(itr->first);
		// }
		// else
		ptrns_by_freq[itr->second].push_back(itr->first);
		itr ++;
	}
	
	double covered_ratio = 0.0;
	bool dom_ratio_reached = false;
	double cut_off_thresh = MIN(0.97, 1 - 3.0 / total_dist_vals);
	cut_off_thresh = MAX(0.7, cut_off_thresh);
	// cerr << "cut_off_thresh = " << cut_off_thresh << endl;
	for (freq_itr=ptrns_by_freq.rbegin(); freq_itr!=ptrns_by_freq.rend(); ++freq_itr){
	// while(freq_itr != ptrns_by_freq.end()){
		covered_ratio += (double)(freq_itr->first * freq_itr->second.size()) / (double)(total_dist_vals);
		// cout << "Covered ratio = " << covered_ratio << endl;
		if(covered_ratio < cut_off_thresh){
			for (size_t i = 0; i < freq_itr->second.size(); i ++){
				dom_ptrns[freq_itr->second[i]] = true;
			}
		}
		else if(!dom_ratio_reached){
			for (size_t i = 0; i < freq_itr->second.size(); i ++){
				dom_ptrns[freq_itr->second[i]] = true;
			}
			dom_ratio_reached = true;
		}
		else{
			for (size_t i = 0; i < freq_itr->second.size(); i ++){
				if ((freq_itr->first >= 4) || ((freq_itr->second.size() * freq_itr->first) >= 4))
					dom_ptrns[freq_itr->second[i]] = true;
				else
					dom_ptrns[freq_itr->second[i]] = false;
			}
		}
		// freq_itr++;
	}
}








