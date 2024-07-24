from error_generator.strategies.utilities.apply_function import Apply_Function


class Error_Generator:

    def error_generator(self, method_gen,selector,percentage,dataset,mute_column):

       
        list_selected_value, number_change =selector.list_selected(dataset,percentage,mute_column)

        dataset = Apply_Function.apply_function(self,number_change=number_change, list_selected_value=list_selected_value,method=method_gen, dataset=dataset)
        return dataset



