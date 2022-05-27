# NADEEF constraints

We leveraged both functional dependencies and denial constraints to detect errors with NADEEF. 
Here is the list of constraints that we leveraged for each dataset:

## Beers:
The following denial constraints already achieve 100% F1-score:
```
rules.append(UDF('ibu', 'value.equals("N/A")'))
rules.append(UDF('abv', '(value != null && !isNumeric(value))'))
rules.append(UDF('city', '((String)tuple.get("state") == null)'))
rules.append(UDF('state', '(value == null)'))
```

## Address:
```
rules.append(UDF('state', 'value != null && value.length() != 2'))
rules.append(UDF('zip', '(value != null && value.length() != 5)'))
rules.append(UDF('ssn', '(value != null && !isNumeric(value))'))
rules.append(UDF('city', 'value != null && value.equals("SAN")'))
rules.append(UDF('city', 'value != null && value.equals("SANTA")'))
rules.append(UDF('city', 'value != null && value.equals("LOS")'))
rules.append(UDF('city', 'value != null && value.equals("EL")'))
rules.append(UDF('city', 'value != null && value.equals("NORTH")'))
rules.append(UDF('city', 'value != null && value.equals("PALM")'))
rules.append(UDF('city', 'value != null && value.equals("WEST")'))
```

The following functional dependencies only lowered the F1-score. Therefore, we did not use them:
```
rules.append(FD(Set(["ZIP"]), "State"))
rules.append(FD(Set(["Address"]), "State"))
```

## Flights:
```
rules.append(UDF('sched_dep_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('act_dep_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('sched_arr_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('act_arr_time', 'value == null || (value != null && value.length() > 10)'))

rules.append(FD(Set(["sched_arr_time", "sched_dep_time"]), "act_dep_time"))
```
Furthermore, among others, we tried the following functional dependencies but none of them increased the F1-score:
```
rules.append(FD(Set(["flight"]), "act_arr_time"))
rules.append(FD(Set(["flight"]), "sched_arr_time"))
rules.append(FD(Set(["flight"]), "act_dep_time"))
rules.append(FD(Set(["flight"]), "sched_dep_time"))
rules.append(FD(Set(["act_arr_time", "sched_arr_time"]), "act_dep_time"))
rules.append(FD(Set(["act_arr_time", "sched_arr_time"]), "sched_dep_time"))
rules.append(FD(Set(["act_arr_time", "act_dep_time"]), "sched_arr_time"))
rules.append(FD(Set(["act_arr_time", "act_dep_time"]), "sched_dep_time"))
rules.append(FD(Set(["act_arr_time", "sched_dep_time"]), "sched_arr_time"))
rules.append(FD(Set(["act_arr_time", "sched_dep_time"]), "act_dep_time"))
rules.append(FD(Set(["act_dep_time", "sched_arr_time"]), "act_arr_time"))
rules.append(FD(Set(["act_dep_time", "sched_arr_time"]), "sched_dep_time"))
rules.append(FD(Set(["sched_arr_time", "sched_dep_time"]), "act_arr_time"))
```

## Hospital:
```
rules.append(UDF('provider_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('zip_code', '(value != null && !isNumeric(value))'))
rules.append(UDF('phone_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('emergency_service', '!(value.equals("Yes") || value.equals("No"))'))
rules.append(UDF('state', '!(value.equals("AL") || value.equals("AK"))'))
```
Furthermore, among others, we tried the following functional dependencies but none of them increased the F1-score:
```
rules.append(FD(Set(["phone_number"]), "zip_code"))
rules.append(FD(Set(["phone_number"]), "city"))
rules.append(FD(Set(["phone_number"]), "state"))
rules.append(FD(Set(["zip_code"]), "city"))
rules.append(FD(Set(["zip_code"]), "state"))
rules.append(FD(Set(["measure_code"]), "measure_name"))
rules.append(FD(Set(["measure_code"]), "condition"))
rules.append(FD(Set(["measure_code", "provider_number"]), "stateavg"))
rules.append(FD(Set(["measure_code", "state"]), "stateavg"))
```

## Movies:
```
rules.append(UDF('Year', 'value != null && value.length() != 4'))
rules.append(UDF('RatingValue', 'value != null && value.length() != 3'))
rules.append(UDF('Id', 'value != null && value.length() != 9'))
rules.append(UDF('Duration', 'value != null && value.length() > 7'))
```
Furthermore, among others, we tried the following functional dependencies but none of them increased the F1-score:
```
rules.append(FD(Set(["Cast", "Duration"]), "Actors")) #0
rules.append(FD(Set(["Description", "Release_Date"]), "Country"))
rules.append(FD(Set(["Name", "Year"]), "Language"))
```

## Restaurants:
No constraints. For instance, the following FDs result in 0% F1-score:
```
rules.append(FD(Set(["city"]), "state"))
rules.append(FD(Set(["zipcode"]), "state"))
```

## Citations:
```
rules.append(UDF('article_jissue', 'value == null'))
rules.append(UDF('article_jvolumn', 'value == null'))
rules.append(FD(Set(['jounral_abbreviation']), 'journal_issn'))
```

## Salary:
```
rules.append(UDF('totalpay', 'Double.parseDouble(value) < 0'))
```






