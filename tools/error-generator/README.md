# Error Generator
A python library to generate highly realistic errors
# Roadmap
this project trying to add several types of error to the dataset.
in the following, you can find the list of the ways that you can inject your dataset

- typos base on keyboards
    - Duplicate the character
    - Delete the character
    - Shift the character one keyboard space

- typos base on [butter-fingers](https://github.com/Decagon/butter-fingers)
    - A python library to generate highly realistic typos (fuzz-testing)
- explicit missing value
   - randomly one value will remove
- implicit missing value:
   - one of median or mode of the active domain, randomly pick and replace with the selected value
- Random Active domain
   - randomly one value from active domain replace with the selected value
- Similar based Active domain
    - the most similar value from the active domain will pick and replace with the selected value 
- White noise (min=0, var=1) 
    - white noise added to the selected value(for string value the noise add to asci code of them)
- gaussian noise:
    - the Gaussian noise added to the selected value (for string value the noise add to asci code of them).
    in this method, you can specify the noise rate as well.
    
# Installation
run the following command in the project directory.
this command installs all packages that the error generator need them for its job.
```
python setup.py install
```

# Application
this project has python interface that helps the user to run it over dataset for injecting the data.

in the ``` user_interface.py ``` you can specify the method that has been explained above and the selector for selecting the columns and cells.

For using the Error generator as part of your project please check [here](https://github.com/BigDaMa/error-generator/blob/master/Help.ipynb)

# Example & test

for using the methods of error generator is needed to make an instance of the methods and selectors and feed the error_generator with them.
``` 
mymethod=one_of_the_methods()
myselector=one_of_the_selectors()
```
```
mygen=Error_Generator()
new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=50,dataset=dataset)
```
in the end, you can save the output in the .csv format in the output folder.

```
Read_Write.write_csv_dataset("./outputs/{}.csv".format(mymethod.name), new_dataset)
```
for more explanation and test of each methods please check the test folder

# output 
the dirty dataset will create in out folder
