class Apply_Function(object):
    def __init__(self):
        pass
    def apply_function(self,number_change,list_selected_value,method,dataset):
        print("---------Change according to {} method ---------------\n".format(method.name))

        for i in range(number_change):
            
            #run(row,col,value,dataset)
            result = method.run(list_selected_value[i][0],list_selected_value[i][1],list_selected_value[i][2],dataset)
            dataset[list_selected_value[i][0]][list_selected_value[i][1]] = result
            print("row: {} col: {} : '{}' changed to '{}'  ".format(list_selected_value[i][0], list_selected_value[i][1],list_selected_value[i][2], result))
 
        return dataset