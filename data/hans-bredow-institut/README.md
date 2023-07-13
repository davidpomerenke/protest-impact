German Local Protest News (GLPN) dataset
----------------------------------------

This dataset contains excerpts from newspaper articles of four German local newspapers labelled for relevancy in protest event analysis.

It can be used to train machine learning models to detect news articles containing mentions of protest event for political analysis.

For using a model trained on this data, it is recommended to preprocess new data in similar ways like this dataset.

To retrieve the excerpts, we the following steps have been taken:
1. split articles into sentences
2. tag sentences that match the following regular expression: protest_regex = re.compile(r'protest|versamm|demonstr|kundgebung|kampagne|soziale bewegung|hausbesetz|streik|unterschriftensammlung|hasskriminalität|unruhen|aufruhr|aufstand|boykott|riot|aktivis|widerstand|mobilisierung|petition|bürgerinitiative|bürgerbegehren|aufmarsch', re.UNICODE | re.IGNORECASE)
3. tag sentences predecessing or succeeding tagged sentences
4. concatenate all tagged sentences to the excerpt.

See the following code on github for an example:
* https://github.com/Leibniz-HBI/protest-event-analysis/blob/main/utils.py contains the function "reformat_df" that preprocesses a column named "text" of a given dataframe
* https://github.com/Leibniz-HBI/protest-event-analysis/blob/main/task-A_prediction.py contains an example on how to apply a model on new data

Experiments on this dataset are described in the following paper:

> Wiedemann, G., Dollbaum, J. M., Haunss, S., Daphi, P., Meier, L. D. (2022): A Generalized Approach to Protest Event Detection in German Local News, In: Proceedings of the 13th International Conference on Language Resources and Evaluation (LREC 2022). Marseille, France. European Language Resources Association (ELRA).

In case of questions on the dataset, please contact Gregor Wiedemann at the Leibniz-Institute for Media Research (HBI): g.wiedemann@leibniz-hbi.de
