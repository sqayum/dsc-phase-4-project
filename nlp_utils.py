import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import validation_curve
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


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





def _gather_tokens(tokenized_documents, by_document=False):
    for token_list in tokenized_documents:
        if by_document:
            token_list = set(token_list)
        for token in token_list:
            yield token

def get_document_frequencies(corpus, label=None, *, N=None):
    if label is not None:
        corpus_tokens = corpus.loc[corpus["label"] == label, "tokens"].reset_index(drop=True)
    else:
        corpus_tokens = corpus.tokens
        label = "ALL"

    if isinstance(corpus_tokens.values[0], str):
        corpus_tokens = corpus_tokens.apply(str.split)

    gathered_tokens = _gather_tokens(corpus_tokens, by_document=True)
    frequency_df = pd.DataFrame(FreqDist(gathered_tokens).items(), columns=['word', 'frequency'])
    frequency_df.sort_values('frequency', inplace=True)

    if N is None:
        N = corpus.shape[0]

    total_num_tokens = frequency_df["frequency"].sum()
    tokens = frequency_df["word"].tail(N)
    normalized_frequencies = [round((frequency / total_num_tokens), 3) for frequency in frequency_df["frequency"].tail(N)]
    return {token: frequency for token, frequency in zip(tokens, normalized_frequencies)}


def plot_document_frequencies(corpus, label=None, *, N=20, figsize=None):
    frequency_dict = get_document_frequencies(corpus, label, N=N)
    if figsize is None:
        fig, ax = plt.subplots(figsize=(15,15))
    else:
        fig, ax = plt.subplots(figsize=figsize)
    tokens = list(frequency_dict.keys())
    normalized_frequencies = list(frequency_dict.values())
    ax.barh(tokens, normalized_frequencies)
    ax.set(title=f'Normalized Document Frequencies ({label})')
    fig.tight_layout()


def plot_label_frequencies(corpus):
    label_counts = corpus.groupby("label").count()["tokens"]
    label_frequencies = label_counts.apply(lambda count: count / label_counts.sum())

    fig, ax = plt.subplots(figsize=(8,7))
    ax.bar(label_frequencies.index, label_frequencies.values, width=0.5)
    ax.set(title="Label Frequencies (Normalized)", ylim=(0,0.5));


def average_token_length(corpus):
    if isinstance(corpus.tokens[0], str):
        corpus.tokens = corpus.tokens.apply(str.split)
    return round(corpus["tokens"].map(len).mean(), 2)


def get_corpus_stop_words(corpus, *,  threshold):
    positive_frequency_dict = get_document_frequencies(corpus, "POSITIVE")
    positive_stop_words = {word for word in positive_frequency_dict if positive_frequency_dict[word] >= threshold}

    neutral_frequency_dict = get_document_frequencies(corpus, "NEUTRAL")
    neutral_stop_words = {word for word in neutral_frequency_dict if neutral_frequency_dict[word] >= threshold}

    negative_frequency_dict = get_document_frequencies(corpus, "NEGATIVE")
    negative_stop_words = {word for word in negative_frequency_dict if negative_frequency_dict[word] >= threshold}

    return (positive_stop_words & neutral_stop_words & negative_stop_words)



def plot_confusion_matrices(y_train_true, y_train_pred,
                             y_validate_true, y_validate_pred,
                             *,
                             labels=None,
                             sample_weights=None,
                             normalize=None,
                             cbar=False,
                             figsize=None):


    cm_train = confusion_matrix(y_train_true, y_train_pred, sample_weight=sample_weights, normalize=normalize)
    cm_validate = confusion_matrix(y_validate_true, y_validate_pred, normalize=normalize)

    cm_train_values = [f"{round(value, 2):.2f}" for value in cm_train.flatten()]
    cm_train_labels = np.asarray(cm_train_values).reshape(cm_train.shape[0], cm_train.shape[1])

    cm_validate_values = [f"{round(value, 2):.2f}" for value in cm_validate.flatten()]
    cm_validate_labels = np.asarray(cm_validate_values).reshape(cm_validate.shape[0], cm_validate.shape[1])

    if figsize is None:
        figsize = (16, 7)

    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 19

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    gs.update(wspace=0.4)

    ax1 = fig.add_subplot(gs[0,0])
    sns.heatmap(cm_train, annot=cm_train_labels, fmt="", cmap="Blues", cbar=cbar, xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set(title="Confusion Matrix (Train)", ylabel = 'True Label', xlabel = 'Predicted Label')

    ax2 = fig.add_subplot(gs[0,1])
    sns.heatmap(cm_validate, annot=cm_validate_labels, fmt="", cmap="Oranges", cbar=cbar, xticklabels=labels, yticklabels=labels, ax=ax2)

    ax2.set(title="Confusion Matrix (Validation)", ylabel = 'True Label', xlabel = 'Predicted Label')

def plot_validation_curve(estimator, X_train, y_train, *, param_name, param_range, scoring, scoring_label, fit_params=None, cv=5, semilogx=False, n_jobs=-1):
    from sklearn.model_selection import validation_curve

    estimator_name = str(estimator)

    train_scores, test_scores = validation_curve(estimator, X_train, y_train,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 scoring=scoring,
                                                 fit_params=fit_params,
                                                 n_jobs=n_jobs)

    avg_train_scores = np.array([np.average(train_scores[i,:]) for i in range(train_scores.shape[0])])
    avg_test_scores = np.array([np.average(test_scores[i,:]) for i in range(test_scores.shape[0])])

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 18
    fig, ax = plt.subplots(figsize=(16,7))
    if not semilogx:
        ax.plot(param_range, avg_train_scores, param_range, avg_test_scores, param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')
    else:
        ax.semilogx(param_range, avg_train_scores, param_range, avg_test_scores, param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')
    ax.set(title=f'Validation Metrics vs {param_name} [{estimator_name}]',
        xlabel=param_name,
        ylabel=scoring_label,
        ylim=(0,1))
    ax.legend(['Train', 'Validate', 'Deviation'])
    plt.show()

def _scale_bar_width(ax, factor):
    from math import isclose
    sorted_patches = sorted(ax.patches, key=lambda x: x.get_x())
    for i, patch in enumerate(sorted_patches):
        current_width = patch.get_width()

        updated_current_width = factor * current_width
        patch.set_width(updated_current_width)

        if i == len(sorted_patches) - 1:
            return

        current_x = patch.get_x()
        next_x = sorted_patches[i+1].get_x()
        if isclose(current_x+current_width, next_x, rel_tol=1e-7, abs_tol=1e-7):
            patch.set_x(next_x - updated_current_width)

def _get_validation_metric(train_metrics_dict , validate_metrics_dict, *, score_name, score_label, target_names):
    train_metric_df = pd.DataFrame(train_metrics_dict).loc[score_name, target_names]
    train_metric_df.name = f"Train (Accuracy = ${round(train_metrics_dict['accuracy'], 3)}$)"

    validate_metric_df = pd.DataFrame(validate_metrics_dict).loc[score_name,target_names]
    validate_metric_df.name = f"Validate (Accuracy = ${round(validate_metrics_dict['accuracy'], 3)}$)"

    return pd.concat([train_metric_df, validate_metric_df], axis=1).stack().reset_index().rename(columns={'level_0': 'Label', 'level_1': 'Dataset', 0: score_label})


def plot_validation_metrics(y_train, y_train_pred, y_validate, y_validate_pred, *, score_names, target_names, estimator_label, sample_weight):

    N = len(score_names)
    fig, axes = plt.subplots(figsize=(15,N*6), nrows=N)

    train_metrics_dict = classification_report(y_train, y_train_pred, target_names=target_names, sample_weight=sample_weight, output_dict=True)
    validate_metrics_dict = classification_report(y_validate, y_validate_pred, target_names=target_names, output_dict=True)

    for i, score_name in enumerate(score_names):
        if '-' in score_name:
            score_label = '-'.join([x.capitalize() for x in score_name.split('-')])
        else:
            score_label = score_name.capitalize()
        score_df = _get_validation_metric(train_metrics_dict , validate_metrics_dict, score_name=score_name, score_label=score_label, target_names=target_names)
        sns.barplot(x=score_df["Label"], y=score_df[score_label], hue=score_df["Dataset"], ax=axes[i])
        axes[i].set(xlabel=None, ylim=(0,1))
        _scale_bar_width(axes[i], 0.55)
        axes[i].yaxis.set_major_locator(MultipleLocator(base=0.1))
        if i == 0:
            axes[i].set(title=f"Validation Metrics by Label [{estimator_label}]")
            axes[i].legend(loc="best")
        else:
            axes[i].get_legend().remove()

def get_classification_metrics(y_true, y_pred, *, target_names, sample_weight=None):
    metrics_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, sample_weight=sample_weight, output_dict=True)).loc["precision":"f1-score", ["NEGATIVE", "NEUTRAL", "POSITIVE", "weighted avg"]].apply(lambda x: round(x, 3))
    metrics_df.index = ["Precision", "Recall", "F1-Score"]
    metrics_df.columns = ["NEGATIVE", "NEUTRAL", "POSITIVE", "Weighted Average"]
    return metrics_df