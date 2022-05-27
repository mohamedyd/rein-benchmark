/**************************************
 **** 2017-4-23      ******************
 **** Abdulhakim Qahtan ****************
 **** aqahtan@hbku.edu.qa ****************
 ***************************************/


#include "csv_reader.h"



bool CSV_READER::check_file_type(std::string const & complete_name, std::string const & extension) 
{
  if (extension.size() > complete_name.size()) return false;
  string str = '.' + extension;  
  return std::equal(str.rbegin(), str.rend(), complete_name.rbegin());
}
// ================ Get the number of columns =====================
long CSV_READER::csv_read_row(std::istream &in, std::vector<std::string> &row) {
  std::stringstream ss;
  bool inquotes = false;
  long number_of_cols = 0;
  while(in.good()) 
  {
    char c = in.get();
    if (!inquotes && c == '"') //beginquotechar
      inquotes = true;
    else if (inquotes && c == '"') //quotechar
    {
      if ( in.peek() == '"') //2 consecutive quotes resolve to 1
        ss << (char)in.get();
      else //endquotechar
        inquotes = false;
    }
    else if (!inquotes && c == field_delimiter) //end of field
    {
      string temp_str = ss.str();
      number_of_cols ++;
      row.push_back(temp_str);
      ss.str("");
    }
    else if (!inquotes && (c == '\r' || c == '\n'))
    {
      if (in.peek() == '\n')  in.get();
      string temp_str = ss.str();
      number_of_cols ++;
      row.push_back(temp_str);
      return number_of_cols;
    }
    else
    {
      ss << c;
    }
  }
  return number_of_cols;
}

// =============================================================================

long CSV_READER::get_number_of_rows(std::istream &in) {
  std::string line;  
  long number_of_rows = 0;  
  while (std::getline(in, line)) {
      number_of_rows ++;
  }
  return number_of_rows;
}


// ========== Read Data to Create a Placeholder for the Output of the Data Cleaning Tools ============
Table CSV_READER::read_csv_file(string file_name) {
      long num_of_cols, num_of_rows = 0, num_skipped = 0;
      // check if the file is comma separated file or not
      if (!check_file_type(file_name, "csv")) {
        cerr << "Not a csv file ..........(" << file_name << ")\n";
        exit(0);
      }
      // === = Open the file and create the results placeholder === == ===
      ifstream in(file_name, ios::in);
      if (!in.good())
      {
        cerr << "Unable to open the csv input file : <" << file_name << ">\n";
        exit(0);
      }
      doubleVecStr data;
      vector<string> header;
      vector<string> row;
      num_of_cols = csv_read_row(in, header); 
     for (long k = 0; k < num_of_cols; k++)
           trim(header[k]);
//          cerr << endl;
      while (in.good())
      {
        long nc = csv_read_row(in, row);
        if (nc == num_of_cols) {
            for (long kk = 0; kk < (long)row.size(); kk++)
                trim(row[kk]);
          data.push_back(row);
          num_of_rows ++;
          for (long k = 0; k < num_of_cols; k++)
            row.pop_back();
        }
        else 
          num_skipped ++;
      }
      Table T(file_name, num_of_rows, num_of_cols, header, data);
      // cerr << "Skipped records = " << num_skipped << endl;
  return T;
}

// ======================================================================

void CSV_READER::display_table(const Table &T)
{
  long i, S = T.number_of_rows;
  for (i = 0; i < (long)T.data[0].size(); i++)
    cout << T.header[i] << ',';
  cout << endl;
  for (long j = 0; j < (long)T.data.size(); j++)
  {
    for (i = 0; i < (long)T.data[0].size(); i++)
      cout << T.data[j][i] << ',';
    cout << endl; 
  }
}
