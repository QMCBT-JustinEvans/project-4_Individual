from sklearn.metrics import classification_report, confusion_matrix

def print_class_metrics(actuals, predictions):

    """
    This Function was adapted and slightly altered 
    from original code provided by Codeup instructor Ryan McCall.
    It provides classification metrics using confusion matrix data.
    """
        
    TN, FP, FN, TP = confusion_matrix(actuals, predictions).ravel()
    
    ALL = TP+TN+FP+FN
    negative_cases = TN+FP
    positive_cases = TP+FN
    
    accuracy = (TP+TN)/ALL
    print(f"Accuracy: {accuracy}")

    true_positive_rate = TP/(TP+FN)
    print(f"True Positive Rate: {true_positive_rate}")

    false_positive_rate = FP/(FP+TN)
    print(f"False Positive Rate: {false_positive_rate}")

    true_negative_rate = TN/(TN+FP)
    print(f"True Negative Rate: {true_negative_rate}")

    false_negative_rate = FN/(FN+TP)
    print(f"False Negative Rate: {false_negative_rate}")

    precision = TP/(TP+FP)
    print(f"Precision: {precision}")

    recall = TP/(TP+FN)
    print(f"Recall: {recall}")

    f1_score = 2*(precision*recall)/(precision+recall)
    print(f"F1 Score: {f1_score}")

    support_pos = TP+FN
    print(f"Support (0): {support_pos}")

    support_neg = FP+TN
    print(f"Support (1): {support_neg}")
    
    # this will return the Series header for y if defined by target= 
        # when conducting split but throws an error if not defined.
    #print(f"Target Feature: {target}, is set for Positive")
    # y_validate.name or y_validate.column

def print_confusion_matrix(actuals, predictions):
    """
    This function returns the sklearn confusion matrix with a helpful visual
    and then returns the classification report.
    """
    print('sklearn Confusion Matrix: (prediction_col, actual_row)')
    print('                          (Negative_first, Positive_second)')
    print(confusion_matrix(actuals, predictions))
    print('                       :--------------------------------------:')
    print('                       | pred Negative(-) | pred Positive (+) |')
    print(' :---------------------:------------------:-------------------:')
    print(' | actual Negative (-) |        TN        |    FP (Type I)    |')
    print(' :---------------------:------------------:-------------------:')
    print(' | actual Positive (+) |   FN (Type II)   |         TP        |')
    print(' :---------------------:------------------:-------------------:')
    print()
    print(classification_report(actuals, predictions))