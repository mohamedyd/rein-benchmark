 /**************************************
 **** 2017-4-23       ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/

#ifndef _CSV_READER_
#define _CSV_READER_

#include "common.h"
#include "Table.h"

#define MAX_CSV_FILE_SIZE 1024*1024*1024		//Set max file size to 1GB

class CSV_READER {
private:
  	const char field_delimiter = ',';
  	long filesize(const char* filename);	
 	long csv_read_row(std::istream &in, std::vector<std::string> &row);
 	long get_number_of_rows(std::istream &in);
	long get_table(const string &filepath, vector<string> &header, doubleVecStr &data);
public:
	CSV_READER() {}
	// Table T;
	Table read_csv_file(string file_name);
	bool check_file_type(std::string const & value, std::string const & ending);
	void display_table(const Table &T);
};
#endif