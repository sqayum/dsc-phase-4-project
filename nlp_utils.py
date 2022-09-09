import numpy as np
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import validation_curve
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from functools import reduce
import re


CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "'cause": "because",
    "cha": "you",
    "coulda": "could have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dunno": "do not know",
    "gimme": "give me",
    "gonna": "going to",
    "gotta": "got to",
    "gotcha": "got you",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "imma": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "kinda": "kind of",
    "lemme": "let me",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "not've": "not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "outta": "out of",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shoulda": "should have",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wanna": "want to",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "woulda": "would have",
    "would've": "would have",
    "wouldn't": "would not",
    "tryna": "trying to",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "u": "you",
    "ya": "you"}

CONVERSATIONAL_ABBREVIATION_MAP = {
    "afaik": "as far as i know",
    "bc": "because",
    "bfn": "bye for now",
    "brb": "be right back",
    "btw": "by the way",
    "dm": "direct message",
    "dyk": "did you know",
    "fomo": "fear of missing out",
    "fb": "facebook",
    "fml": "fuck my life",
    "fr": "for real",
    "ftf": "face to face",
    "ftl": "for the loss",
    "ftw": "for the win",
    "fwd": "forward",
    "fwiw": "for what it is worth",
    "fyi": "for your information",
    "gtg": "got to go",
    "gtr": "got to run",
    "hifw": "how i feel when",
    "hmb": "hit me back",
    "hmu": "hit me up",
    "hth": "hope that helps",
    "idc": "i do not care",
    "idk": "i do not know",
    "ikr": "i know right",
    "ily": "i love you",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "irl": "in real life",
    "jk": "just kidding",
    "lmao": "laughing my ass off",
    "lmk": "let me know",
    "lol": "laughing out loud",
    "nbd": "no big deal",
    "nm": "not much",
    "nfw": "no fucking way",
    "nsfw": "not safe for work",
    "nvm": "nevermind",
    "omfg": "oh my fucking God",
    "omg": "oh my God",
    "omw": "on my way",
    "ppl": "people",
    "rly": "really",
    "rofl": "rolling on the floor laughing",
    "sfw": "safe for work",
    "smh": "shaking my head",
    "stfu": "shut the fuck up",
    "tbh": "to be honest",
    "tfw": "that feeling when",
    "tgif": "thank God its Friday",
    "tmi": "too much information",
    "tldr": "too long did not read",
    "wbu": "what about you",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "ty": "thank you",
    "txt": "text",
    "yolo": "you only live once",
    "yw": "your welcome",
    "zomg": "oh my God"}

TECHNICAL_ABBREVIATION_MAP = {
    "DM": "direct message",
    "CT": "cuttweet",
    "RT": "retweet",
    "MT": "modified tweet",
    "HT": "hat tip",
    "CC": "carbon-copy",
    "CX": "correction",
    "FB": "Facebook",
    "LI": "LinkedIn",
    "YT": "YouTube"}


def print_completion_message(*, start_msg=None, end_msg=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal start_msg
            nonlocal end_msg
            arg_str = ', '.join([str(arg) for arg in args] + [f"{key}={value}" for key, value in kwargs.items()])
            if start_msg is None:
                start_msg = f"EXECUTING <{func.__name__}({arg_str})> ......"
            print(f"{start_msg} ......", end='')
            result = func(*args, **kwargs)
            if end_msg is None:
                end_msg = f" COMPLETE"
            print(end_msg)
            return result
        return wrapper
    return decorator



def _gather_tokens(tokenized_documents, *, by_document=False):
    for token_list in tokenized_documents:
        if by_document:
            token_list = set(token_list)
        for token in token_list:
            yield token


def _get_document_frequencies(corpus,
                                   *,
                                   document_col,
                                   label_col,
                                   label_name=None,
                                   N=None):

    if label_name is None:
        tokenized_documents = corpus[document_col]
    else:
        tokenized_documents = corpus.loc[corpus[label_col] == label_name, document_col].reset_index(drop=True)

    if isinstance(tokenized_documents.values[0], str):
        tokenized_documents = tokenized_documents.apply(str.split)

    gathered_tokens = _gather_tokens(tokenized_documents, by_document=True)
    frequency_df = pd.DataFrame(FreqDist(gathered_tokens).items(), columns=['word', 'frequency'])
    frequency_df.sort_values('frequency', inplace=True)
    if N is None:
        N = corpus.shape[0]
    total_num_tokens = frequency_df["frequency"].sum()
    tokens = frequency_df["word"].tail(N)
    normalized_frequencies = [round((frequency / total_num_tokens), 3) for frequency in frequency_df["frequency"].tail(N)]

    return {token: frequency for token, frequency in zip(tokens, normalized_frequencies)}


def plot_document_frequencies(corpus,
                                  *,
                                  document_col,
                                  label_col,
                                  label_name=None,
                                  N=20,
                                  figsize=None,
                                  filepath=None):

    frequency_dict = _get_document_frequencies(corpus,
                                                      document_col=document_col,
                                                      label_col=label_col,
                                                      label_name=label_name,
                                                      N=N)
    if figsize is None:
        fig, ax = plt.subplots(figsize=(15,15))
    else:
        fig, ax = plt.subplots(figsize=figsize)
    tokens = list(frequency_dict.keys())
    normalized_frequencies = list(frequency_dict.values())
    ax.barh(tokens, normalized_frequencies)
    ax.set(title=f'Normalized Document Frequencies ({label_name})')
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath)


def plot_label_frequencies(corpus,
                              *,
                              document_col,
                              label_col,
                              filepath=None):

    label_counts = corpus.groupby(label_col).count()[document_col]
    label_frequencies = label_counts.apply(lambda count: count / label_counts.sum())

    fig, ax = plt.subplots(figsize=(8,7))
    ax.bar(label_frequencies.index, label_frequencies.values, width=0.5)
    ax.set(title="Label Frequencies (Normalized)", ylim=(0,0.5));

    if filepath is not None:
        fig.savefig(filepath)


def average_token_length(corpus, *, document_col):
    if isinstance(corpus[document_col][0], str):
        tokenized_documents = corpus[document_col].apply(str.split)
    else:
        tokenized_documents = corpus[document_col]
    return round(tokenized_documents.map(len).mean(), 2)



def plot_confusion_matrices(y_train_true,
                                y_train_pred,
                                y_validate_true,
                                y_validate_pred,
                                *,
                                labels=None,
                                sample_weights=None,
                                normalize=None,
                                mode="validate",
                                cbar=False,
                                figsize=None,
                                filepath=None):

    cm_train = confusion_matrix(y_train_true,
                                     y_train_pred,
                                     sample_weight=sample_weights,
                                     normalize=normalize)

    cm_validate = confusion_matrix(y_validate_true,
                                        y_validate_pred,
                                        normalize=normalize)

    cm_train_values = [f"{round(value, 2):.2f}" for value in cm_train.flatten()]
    cm_train_labels = np.asarray(cm_train_values).reshape(cm_train.shape[0], cm_train.shape[1])

    cm_validate_values = [f"{round(value, 2):.2f}" for value in cm_validate.flatten()]
    cm_validate_labels = np.asarray(cm_validate_values).reshape(cm_validate.shape[0], cm_validate.shape[1])

    if figsize is None:
        figsize = (13, 5)

    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 19

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    gs.update(wspace=0.4)

    ax1 = fig.add_subplot(gs[0,0])
    sns.heatmap(cm_train,
                   annot=cm_train_labels,
                   fmt="",
                   cmap="Blues",
                   cbar=cbar,
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=ax1)

    ax1.set(title="Confusion Matrix (Train)",
             ylabel ='True Label',
             xlabel ='Predicted Label')

    ax2 = fig.add_subplot(gs[0,1])
    sns.heatmap(cm_validate,
                   annot=cm_validate_labels,
                   fmt="",
                   cmap="Oranges",
                   cbar=cbar,
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=ax2)

    if mode == "validate":
        title = "Confusion Matrix (Validation)"
    elif mode == "test":
        title = "Confusion Matrix (Test)"

    ax2.set(title=title,
             ylabel ='True Label',
             xlabel ='Predicted Label')

    if filepath is not None:
        fig.savefig(filepath)


def plot_validation_curve(estimator,
                             X_train,
                             y_train,
                             *,
                             param_name,
                             param_range,
                             scoring,
                             scoring_label,
                             fit_params=None,
                             cv=5,
                             semilogx=False,
                             n_jobs=-1,
                             figsize=(16,12),
                             filepath=None):

    estimator_name = str(estimator)

    train_scores, test_scores = validation_curve(estimator,
                                                       X_train,
                                                       y_train,
                                                       param_name=param_name,
                                                       param_range=param_range,
                                                       scoring=scoring,
                                                       fit_params=fit_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       verbose=2)

    avg_train_scores = np.array([np.average(train_scores[i,:]) for i in range(train_scores.shape[0])])
    avg_test_scores = np.array([np.average(test_scores[i,:]) for i in range(test_scores.shape[0])])

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 18

    fig, (ax1,ax2) = plt.subplots(figsize=figsize, nrows=2, ncols=1, sharex=True)
    if not semilogx:
        ax1.plot(param_range, avg_train_scores, param_range, avg_test_scores)
        ax2.plot(param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')
    else:
        ax1.semilogx(param_range, avg_train_scores, param_range, avg_test_scores)
        ax2.semilogx(param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')

    ax1.set(title=f'{scoring_label} vs {param_name} [{estimator_name}]',
             ylabel=scoring_label)
    ax2.set(xlabel=f'{param_name}',
             ylabel=f'{scoring_label} Deviation')

    ax1.legend(['Train', 'Validate'])

    plt.show()

    if filepath is not None:
        fig.savefig(filepath)


def _scale_bar_width(ax, factor, *, horizontal=False):
    from math import isclose

    if not horizontal:
        sorted_patches = sorted(ax.patches, key=lambda x: x.get_x())
        for i, patch in enumerate(sorted_patches):
            current_width = patch.get_width()

            updated_current_width = factor * current_width
            patch.set_width(updated_current_width)

            if i == len(sorted_patches) - 1:
                return

            current_x = patch.get_x()
            next_x = sorted_patches[i+1].get_x()
            if isclose(current_x + current_width, next_x, rel_tol=1e-7, abs_tol=1e-7):
                patch.set_x(next_x - updated_current_width)
    else:
        sorted_patches = sorted(ax.patches, key=lambda x: x.get_y())
        for i, patch in enumerate(sorted_patches):
            current_width = patch.get_width()

            updated_current_width = factor * current_width
            patch.set_width(updated_current_width)

            if i == len(sorted_patches) - 1:
                return

            current_y = patch.get_y()
            next_y = sorted_patches[i+1].get_y()
            if isclose(current_y + current_width, next_y, rel_tol=1e-7, abs_tol=1e-7):
                patch.set_y(next_y - updated_current_width)


def _get_validation_metric(train_metrics_dict,
                               validate_metrics_dict,
                               *,
                               score_name,
                               score_label,
                               target_names,
                               mode):

    train_metric_df = pd.DataFrame(train_metrics_dict).loc[score_name, target_names]
    train_metric_df.name = f"Train (Accuracy = ${round(train_metrics_dict['accuracy'], 3)}$)"

    validate_metric_df = pd.DataFrame(validate_metrics_dict).loc[score_name, target_names]
    if mode == "validate":
        validate_metric_df.name = f"Validate (Accuracy = ${round(validate_metrics_dict['accuracy'], 3)}$)"
    if mode == "test":
        validate_metric_df.name = f"Test (Accuracy = ${round(validate_metrics_dict['accuracy'], 3)}$)"

    return pd.concat([train_metric_df, validate_metric_df], axis=1).stack().reset_index().rename(columns={'level_0': 'Label', 'level_1': 'Dataset', 0: score_label})


def plot_validation_metrics_by_label(y_train,
                                         y_train_pred,
                                         y_validate,
                                         y_validate_pred,
                                         *,
                                         score_names,
                                         target_names,
                                         estimator_label,
                                         sample_weight,
                                         mode="validate",
                                         figsize=None,
                                         filepath=None):

    N = len(score_names)

    if figsize is None:
        figsize = (11,N*5)
    fig, axes = plt.subplots(figsize=figsize, nrows=N)
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 19

    train_metrics_dict = classification_report(y_train,
                                                    y_train_pred,
                                                    target_names=target_names,
                                                    sample_weight=sample_weight,
                                                    output_dict=True)

    validate_metrics_dict = classification_report(y_validate,
                                                       y_validate_pred,
                                                       target_names=target_names,
                                                       output_dict=True)

    for i, score_name in enumerate(score_names):
        if N == 1:
            axes = [axes]

        if '-' in score_name:
            score_label = '-'.join([x.capitalize() for x in score_name.split('-')])
        else:
            score_label = score_name.capitalize()

        score_df = _get_validation_metric(train_metrics_dict,
                                               validate_metrics_dict,
                                               score_name=score_name,
                                               score_label=score_label,
                                               target_names=target_names,
                                               mode=mode)

        sns.barplot(x=score_df["Label"],
                      y=score_df[score_label],
                      hue=score_df["Dataset"],
                      ax=axes[i])

        axes[i].set(xlabel=None, ylim=(0,1))
        _scale_bar_width(axes[i], 0.55)
        axes[i].yaxis.set_major_locator(MultipleLocator(base=0.1))
        if i == 0:
            if mode == "validate":
                title = f"Validation Metrics by Label [{estimator_label}]"
            elif mode == "test":
                title = f"Test Metrics by Label [{estimator_label}]"
            axes[i].set(title=title)
            axes[i].legend(loc="best", prop={'size': 14}, ncol=2)
        else:
            axes[i].get_legend().remove()

    if filepath is not None:
        fig.savefig(filepath)


def get_classification_metrics(y_true,
                                   y_pred,
                                   *,
                                   target_names,
                                   sample_weight=None,
                                   average_only=False,
                                   filepath=None):

    metrics_dict = classification_report(y_true,
                                             y_pred,
                                             target_names=target_names,
                                             sample_weight=sample_weight,
                                             output_dict=True)

    accuracy = round(metrics_dict["accuracy"], 3)
    num_labels = len(target_names)

    metrics_df = pd.DataFrame(metrics_dict)
    target_names = list(target_names)
    if sample_weight is not None:
        target_names.append("weighted avg")
        metrics_df = metrics_df.loc["precision":"f1-score", target_names].apply(lambda x: round(x, 3))
        metrics_df.rename(columns={"weighted avg": "Weighted Average"}, inplace=True)
    else:
        target_names.append("macro avg")
        metrics_df = metrics_df.loc["precision":"f1-score", target_names].apply(lambda x: round(x, 3))
        metrics_df.rename(columns={"macro avg": "Average"}, inplace=True)
    metrics_df.index = ["Precision", "Recall", "F1-Score"]

    metrics_df.loc["Accuracy", metrics_df.columns[num_labels:]] = accuracy

    if average_only:
        metrics_df = pd.DataFrame(metrics_df["Average"])

    return metrics_df



def plot_history(model_name,
                   history_dict,
                   *,
                   filepath=None):

    training_loss_values = history_dict['loss']
    validation_loss_values = history_dict['val_loss']
    epochs = range(1, len(training_loss_values)+1)

    training_accuracy_values = history_dict['accuracy']
    validation_accuracy_values = history_dict['val_accuracy']
    epochs = range(1, len(training_accuracy_values)+1)

    fig, (ax1, ax2) = plt.subplots(figsize=(15,10), nrows=2, ncols=1, sharex=True)

    ax1.plot(epochs, training_loss_values, 'tab:blue', label='Training Loss')
    ax1.plot(epochs, validation_loss_values, 'tab:orange', label='Validation Loss')
    ax1.set(title=f"History ({model_name})" , ylabel='Crossentropy Loss')

    ax2.plot(epochs, training_accuracy_values, 'tab:blue', label='Training Accuracy')
    ax2.plot(epochs, validation_accuracy_values, 'tab:orange', label='Validation Accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')

    ax1.legend()
    ax2.legend()
    plt.show()

    if filepath is not None:
        fig.savefig(filepath)



def get_english_stopwords():
    stopwords = set()
    with open("data/english_stopwords.txt") as file_iter:
        for word in file_iter.readlines():
            stopwords.add(word.strip())
    return stopwords


def get_corpus_stopwords(corpus,
                             *,
                             document_col,
                             label_col,
                             threshold=0.0001):

    stopwords_by_label = {}
    for label_name in corpus[label_col].unique():
        label_frequency_dict = _get_document_frequencies(corpus,
                                                                 document_col=document_col,
                                                                 label_col=label_col,
                                                                 label_name=label_name)

        label_stopwords = {word for word in label_frequency_dict if label_frequency_dict[word] >= threshold}
        stopwords_by_label[label_name] = label_stopwords
    return reduce(lambda x,y: x&y, stopwords_by_label.values())


def regex_scan(corpus,
                 *,
                 col_name,
                 pattern,
                 flags=None,
                 append_to_corpus=False,
                 new_col_name=None):

    if flags is None:
        pattern = re.compile(pattern)
    else:
        pattern = re.compile(pattern, flags=flags)
    result_col = corpus[col_name].str.contains(pattern, na=False)
    if not append_to_corpus:
        return result_col
    else:
        assert isinstance(new_col_name, str)
        corpus[new_col_name] = result_col
