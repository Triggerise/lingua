# Copyright © 2018-2020 Peter M. Stahl pemistahl@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expressed or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import pandas as pd


def write_comparison_table(data, filename):
    rounded_mean_data = data.mean().round().astype(int)
    rounded_median_data = data.median().round(2)
    rounded_std_data = data.std().round(2)
    column_names = (
        'average-lingua', 'average-tika', 'average-opennlp', 'average-optimaize',
        'single-words-lingua', 'single-words-tika', 'single-words-opennlp', 'single-words-optimaize',
        'word-pairs-lingua', 'word-pairs-tika', 'word-pairs-opennlp', 'word-pairs-optimaize',
        'sentences-lingua', 'sentences-tika', 'sentences-opennlp', 'sentences-optimaize'
    )

    table = """<table>
    <tr>
        <th>Language</th>
        <th colspan="4">Average</th>
        <th colspan="4">Single Words</th>
        <th colspan="4">Word Pairs</th>
        <th colspan="4">Sentences</th>
    </tr>
    <tr>
        <th></th>
        <th>Lingua</th>
        <th>&nbsp;&nbsp;Tika&nbsp;&nbsp;</th>
        <th>OpenNLP</th>
        <th>Optimaize</th>
        <th>Lingua</th>
        <th>&nbsp;&nbsp;Tika&nbsp;&nbsp;</th>
        <th>OpenNLP</th>
        <th>Optimaize</th>
        <th>Lingua</th>
        <th>&nbsp;&nbsp;Tika&nbsp;&nbsp;</th>
        <th>OpenNLP</th>
        <th>Optimaize</th>
        <th>Lingua</th>
        <th>&nbsp;&nbsp;Tika&nbsp;&nbsp;</th>
        <th>OpenNLP</th>
        <th>Optimaize</th>
    </tr>
    """

    for language in data.index:
        language_data = data.loc[language]
        table += "\t<tr>\n\t\t<td>" + language + "</td>\n"

        for column in column_names:
            accuracy_value = language_data.loc[[column]][0]
            if not math.isnan(accuracy_value):
                accuracy_value = int(round(accuracy_value))
                accuracy_str = str(accuracy_value)
            else:
                accuracy_str = "-"

            color = get_square_color(accuracy_value)
            table += "\t\t<td><img src=\"images/" + color + ".png\"> " + accuracy_str + "</td>\n"

        table += "\t</tr>\n"

    table += "\t<tr>\n\t\t<td colspan=\"12\"></td>\n\t</tr>\n"
    table += "\t<tr>\n\t\t<td><strong>Mean</strong></td>\n"

    for column in column_names:
        accuracy_value = rounded_mean_data.loc[[column]][0]
        color = get_square_color(accuracy_value)
        table += "\t\t<td><img src=\"images/" + color + ".png\"> <strong>" + str(accuracy_value) + "</strong></td>\n"

    table += "\t</tr>\n"

    table += "\t<tr>\n\t\t<td colspan=\"12\"></td>\n\t</tr>\n"
    table += "\t<tr>\n\t\t<td>Median</td>\n"

    for column in column_names:
        accuracy_value = rounded_median_data.loc[[column]][0]
        table += "\t\t<td>" + str(accuracy_value) + "</td>\n"

    table += "\t</tr>\n"

    table += "\t<tr>\n\t\t<td>Standard Deviation</td>\n"

    for column in column_names:
        accuracy_value = rounded_std_data.loc[[column]][0]
        table += "\t\t<td>" + str(accuracy_value) + "</td>\n"

    table += "\t</tr>\n"
    table += "</table>"

    with open(filename, 'w') as comparison_table_file:
        comparison_table_file.write(table)


def get_square_color(accuracy_value):
    if math.isnan(accuracy_value):
        return "grey"
    elif 0 <= accuracy_value <= 20:
        return "red"
    elif 21 <= accuracy_value <= 40:
        return "orange"
    elif 41 <= accuracy_value <= 60:
        return "yellow"
    elif 61 <= accuracy_value <= 80:
        return "lightgreen"
    elif 81 <= accuracy_value <= 100:
        return "green"
    else:
        raise ValueError("invalid accuracy value:", accuracy_value)


accuracy_values_data_frame = pd.read_csv(
    filepath_or_buffer='accuracy-reports/aggregated-accuracy-values.csv'
).set_index('language')

accuracy_values_data_frame = accuracy_values_data_frame.reindex(
    sorted(accuracy_values_data_frame.columns),
    axis=1
)

write_comparison_table(accuracy_values_data_frame, filename='ACCURACY_TABLE.md')

print("Accuracy table written successfully")
