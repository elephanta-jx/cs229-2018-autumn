import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.strip().lower().split()

    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    freq_dic = {}
    for msg in messages:
        words = get_words(msg)
        words_set = set(words)
        for word in words_set:
            
            if word in freq_dic:
                freq_dic[word] += 1
            else:
                freq_dic[word] = 1

    res_dic = {}
    i = 0
    for key in freq_dic:
        if freq_dic[key] >= 5:
            res_dic[key] = i
            i += 1
    
    return res_dic

    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    arr = np.zeros((len(messages), len(word_dictionary)))

    for i, msg in enumerate(messages):
        words = get_words(msg)
        for word in words:
            if word_dictionary.get(word) is not None:
                arr[i, word_dictionary[word]] += 1

    return arr

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    y_1 = 0
    for b in labels:
        if b == 1:
            y_1 += 1

    
    phi_1 = np.ones(matrix.shape[-1])
    phi_0 = np.ones(matrix.shape[-1])

    
    for (row, col), elem in np.ndenumerate(matrix):
        if elem > 0 and labels[row]:
            phi_1[col] += 1
        elif elem > 0 and not labels[row]:
            phi_0[col] += 1
    
    # this should be 2 instead of number of words in the dict. I think the answer is wrong since its 2 in the notes
    phi_1 = phi_1 / (y_1 + 2)     
    phi_0 = phi_0 / (len(labels) - y_1 + 2)

    phi_y = y_1 / len(labels)
    
    return (phi_y, phi_0, phi_1)


    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi_y, phi_0, phi_1 = model

    res = np.zeros(matrix.shape[0])

    for r_idx, rows in enumerate(matrix):
                
        log_proc_1 = np.log(phi_y)
        log_proc_0 = np.log((1 - phi_y))
        for i, freq in enumerate(rows):
            if freq > 0:
                log_proc_1 += np.log(phi_1[i])
                log_proc_0 += np.log(phi_0[i])

        ratio = np.exp(log_proc_1 - log_proc_0)
        
        if ratio / (ratio + 1) > 0.5:
            res[r_idx] = 1
        else: 
            res[r_idx] = 0
        
    return res

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_y, phi_0, phi_1 = model

    unsort = np.zeros(len(phi_0), dtype=[('v', np.float64), ('idx', int)])
    for i, p0 in enumerate(phi_0):
        unsort[i] = (np.log(phi_1[i] / p0), i)
    
    sorted = np.sort(unsort, order='v')[-1:-6:-1]

    keys = []
    for elem in sorted:
        find = [k for k, v in dictionary.items() if v == elem[1]]
        keys.append(find[0])

    return keys
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accuracy = []
    for radius in radius_to_consider:
        predict = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy.append(np.mean(predict == val_labels))

    idx = accuracy.index(max(accuracy))
    return radius_to_consider[idx]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
