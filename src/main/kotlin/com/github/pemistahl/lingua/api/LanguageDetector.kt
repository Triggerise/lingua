/*
 * Copyright © 2018-2020 Peter M. Stahl pemistahl@gmail.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expressed or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.pemistahl.lingua.api

import com.github.pemistahl.lingua.api.Language.FRENCH
import com.github.pemistahl.lingua.api.Language.SPANISH
import com.github.pemistahl.lingua.api.Language.UNKNOWN
import com.github.pemistahl.lingua.internal.Alphabet
import com.github.pemistahl.lingua.internal.Constant.MULTIPLE_WHITESPACE
import com.github.pemistahl.lingua.internal.Constant.NUMBERS
import com.github.pemistahl.lingua.internal.Constant.PUNCTUATION
import com.github.pemistahl.lingua.internal.Ngram
import com.github.pemistahl.lingua.internal.TestDataLanguageModel
import com.github.pemistahl.lingua.internal.TrainingDataLanguageModel
import com.github.pemistahl.lingua.internal.util.extension.containsAnyOf
import com.github.pemistahl.lingua.internal.util.extension.incrementCounter
import java.util.SortedMap
import java.util.TreeMap
import java.util.regex.PatternSyntaxException
import kotlin.math.ln

/**
 * Detects the language of given input text.
 */
class LanguageDetector internal constructor(
    internal val languages: MutableSet<Language>,
    internal val minimumRelativeDistance: Double,
    internal val numberOfLoadedLanguages: Int = languages.size
) {
    private val languagesWithUniqueCharacters = languages.filterNot { it.uniqueCharacters.isNullOrBlank() }.asSequence()
    private val alphabetsSupportingExactlyOneLanguage = Alphabet.allSupportingExactlyOneLanguage().filterValues {
        it in languages
    }

    internal val unigramLanguageModels = loadLanguageModels(ngramLength = 1)
    internal val bigramLanguageModels = loadLanguageModels(ngramLength = 2)
    internal val trigramLanguageModels = loadLanguageModels(ngramLength = 3)
    internal val quadrigramLanguageModels = loadLanguageModels(ngramLength = 4)
    internal val fivegramLanguageModels = loadLanguageModels(ngramLength = 5)

    /**
     * Detects the language of given input text.
     *
     * @param text The input text to detect the language for.
     * @return The identified language or [Language.UNKNOWN].
     */
    fun detectLanguageOf(text: String): Language {
        val confidenceValues = computeLanguageConfidenceValues(text)

        if (confidenceValues.isEmpty()) return UNKNOWN
        if (confidenceValues.size == 1) return confidenceValues.firstKey()

        val mostLikelyLanguage = confidenceValues.firstKey()
        val mostLikelyLanguageProbability = confidenceValues.getValue(mostLikelyLanguage)

        val secondMostLikelyLanguage = confidenceValues.filterNot {
            it.key == mostLikelyLanguage
        }.maxByOrNull { it.value }!!.key
        val secondMostLikelyLanguageProbability = confidenceValues.getValue(secondMostLikelyLanguage)

        if (mostLikelyLanguageProbability == secondMostLikelyLanguageProbability) return UNKNOWN
        if ((mostLikelyLanguageProbability - secondMostLikelyLanguageProbability) >= minimumRelativeDistance) {
            return mostLikelyLanguage
        }

        return UNKNOWN
    }

    /**
     * Computes confidence values for every language considered possible for the given input text.
     *
     * The values that this method computes are part of a **relative** confidence metric, not of an absolute one.
     * Each value is a number between 0.0 and 1.0. The most likely language is always returned with value 1.0.
     * All other languages get values assigned which are lower than 1.0, denoting how less likely those languages
     * are in comparison to the most likely language.
     *
     * The map returned by this method does not necessarily contain all languages which the calling instance of
     * [LanguageDetector] was built from. If the rule-based engine decides that a specific language is truly impossible,
     * then it will not be part of the returned map. Likewise, if no ngram probabilities can be found within the
     * detector's languages for the given input text, the returned map will be empty. The confidence value for
     * each language not being part of the returned map is assumed to be 0.0.
     *
     * @param text The input text to detect the language for.
     * @return A map of all possible languages, sorted by their confidence value in descending order.
     */
    fun computeLanguageConfidenceValues(text: String): SortedMap<Language, Double> {
        val values = TreeMap<Language, Double>()
        val cleanedUpText = cleanUpInputText(text)

        if (cleanedUpText.isEmpty() || NO_LETTER.matches(cleanedUpText)) return values

        val words = splitTextIntoWords(cleanedUpText)
        val languageDetectedByRules = detectLanguageWithRules(words)

        if (languageDetectedByRules != UNKNOWN) {
            values[languageDetectedByRules] = 1.0
            return values
        }

        val allProbabilities = mutableListOf<Map<Language, Double>>()
        val unigramCountsOfInputText = mutableMapOf<Language, Int>()
        var languagesSequence = filterLanguagesByRules(words)

        for (i in 1..5) {
            if (cleanedUpText.length < i) continue

            val testDataModel = TestDataLanguageModel.fromText(cleanedUpText, ngramLength = i)
            allProbabilities.add(computeLanguageProbabilities(testDataModel, languagesSequence))
            val languageKeys = allProbabilities.last().keys

            if (languageKeys.isNotEmpty()) {
                languagesSequence = languagesSequence.filter { it in languageKeys }
            }

            if (i == 1) countUnigramsOfInputText(unigramCountsOfInputText, testDataModel, languagesSequence)
        }

        val summedUpProbabilities = sumUpProbabilities(allProbabilities, unigramCountsOfInputText, languagesSequence)
        val highestProbability = summedUpProbabilities.maxByOrNull { it.value }?.value ?: return sortedMapOf()
        val confidenceValues = summedUpProbabilities.mapValues { highestProbability / it.value }
        val sortedByConfidenceValue = compareByDescending<Language> { language -> confidenceValues[language] }
        val sortedByConfidenceValueThenByLanguage = sortedByConfidenceValue.thenBy { language -> language }

        return confidenceValues.toSortedMap(sortedByConfidenceValueThenByLanguage)
    }

    internal fun cleanUpInputText(text: String): String {
        return text.trim().toLowerCase()
            .replace(PUNCTUATION, "")
            .replace(NUMBERS, "")
            .replace(MULTIPLE_WHITESPACE, " ")
    }

    internal fun splitTextIntoWords(text: String): List<String> {
        return if (text.contains(' ')) {
            text.split(' ')
        } else {
            listOf(text)
        }
    }

    internal fun countUnigramsOfInputText(
        unigramCounts: MutableMap<Language, Int>,
        unigramLanguageModel: TestDataLanguageModel,
        languagesSequence: Sequence<Language>
    ) {
        for (language in languagesSequence) {
            for (unigram in unigramLanguageModel.ngrams) {
                val probability = lookUpNgramProbability(language, unigram)
                if (probability > 0) {
                    unigramCounts.incrementCounter(language)
                }
            }
        }
    }

    internal fun sumUpProbabilities(
        probabilities: List<Map<Language, Double>>,
        unigramCountsOfInputText: Map<Language, Int>,
        languagesSequence: Sequence<Language>
    ): Map<Language, Double> {
        val summedUpProbabilities = hashMapOf<Language, Double>()
        for (language in languagesSequence) {
            summedUpProbabilities[language] = probabilities.sumByDouble { it[language] ?: 0.0 }

            if (unigramCountsOfInputText.containsKey(language)) {
                summedUpProbabilities[language] = summedUpProbabilities.getValue(language) /
                    unigramCountsOfInputText.getValue(language)
            }
        }
        return summedUpProbabilities.filter { it.value != 0.0 }
    }

    internal fun detectLanguageWithRules(words: List<String>): Language {
        val totalLanguageCounts = mutableMapOf<Language, Int>()

        for (word in words) {
            val wordLanguageCounts = mutableMapOf<Language, Int>()

            for (character in word.map { it.toString() }) {
                var isMatch = false
                for ((alphabet, language) in alphabetsSupportingExactlyOneLanguage) {
                    if (alphabet.matches(character)) {
                        wordLanguageCounts.incrementCounter(language)
                        isMatch = true
                    }
                }
                if (!isMatch) {
                    when {
                        Alphabet.LATIN.matches(character) ||
                            Alphabet.DEVANAGARI.matches(character) ->
                            languagesWithUniqueCharacters.filter {
                                it.uniqueCharacters?.contains(character) ?: false
                            }.forEach {
                                wordLanguageCounts.incrementCounter(it)
                            }
                    }
                }
            }

            if (wordLanguageCounts.isEmpty()) {
                totalLanguageCounts.incrementCounter(UNKNOWN)
            } else if (wordLanguageCounts.size == 1) {
                val language = wordLanguageCounts.toList().first().first
                if (language in languages) {
                    totalLanguageCounts.incrementCounter(language)
                } else {
                    totalLanguageCounts.incrementCounter(UNKNOWN)
                }
            } else {
                val sortedWordLanguageCounts = wordLanguageCounts.toList().sortedByDescending { it.second }
                val (mostFrequentLanguage, firstCharCount) = sortedWordLanguageCounts[0]
                val (_, secondCharCount) = sortedWordLanguageCounts[1]

                if (firstCharCount > secondCharCount && mostFrequentLanguage in languages) {
                    totalLanguageCounts.incrementCounter(mostFrequentLanguage)
                } else {
                    totalLanguageCounts.incrementCounter(UNKNOWN)
                }
            }
        }

        val unknownLanguageCount = totalLanguageCounts[UNKNOWN] ?: 0
        val filteredLanguageCounts = if (unknownLanguageCount >= (0.5 * words.size)) {
            totalLanguageCounts
        } else {
            totalLanguageCounts.filterNot { it.key == UNKNOWN }
        }

        if (filteredLanguageCounts.isEmpty()) {
            return UNKNOWN
        }
        if (filteredLanguageCounts.size == 1) {
            return filteredLanguageCounts.toList().first().first
        }

        val sortedTotalLanguageCounts = filteredLanguageCounts.toList().sortedByDescending { it.second }
        val (mostFrequentLanguage, firstCharCount) = sortedTotalLanguageCounts[0]
        val (_, secondCharCount) = sortedTotalLanguageCounts[1]

        if (firstCharCount == secondCharCount) {
            return UNKNOWN
        }

        return mostFrequentLanguage
    }

    internal fun filterLanguagesByRules(words: List<String>): Sequence<Language> {
        val detectedAlphabets = mutableMapOf<Alphabet, Int>()

        for (word in words) {
            for (alphabet in Alphabet.values()) {
                if (alphabet.matches(word)) {
                    detectedAlphabets.incrementCounter(alphabet)
                    break
                }
            }
        }

        if (detectedAlphabets.isEmpty()) {
            return languages.asSequence()
        }

        val mostFrequentAlphabet = detectedAlphabets.entries.maxByOrNull { it.value }!!.key
        val filteredLanguages = languages.asSequence().filter { it.alphabets.contains(mostFrequentAlphabet) }
        val languageCounts = mutableMapOf<Language, Int>()

        for (word in words) {
            for ((characters, languages) in CHARS_TO_LANGUAGES_MAPPING) {
                if (word.containsAnyOf(characters)) {
                    for (language in languages) {
                        languageCounts.incrementCounter(language)
                    }
                    break
                }
            }
        }

        val languagesSubset = languageCounts.filterValues { it >= words.size / 2 }.keys

        return if (languagesSubset.isNotEmpty()) {
            filteredLanguages.filter { it in languagesSubset }
        } else {
            filteredLanguages
        }
    }

    internal fun computeLanguageProbabilities(
        testDataModel: TestDataLanguageModel,
        languagesSequence: Sequence<Language>
    ): Map<Language, Double> {
        val probabilities = hashMapOf<Language, Double>()
        for (language in languagesSequence) {
            probabilities[language] = computeSumOfNgramProbabilities(language, testDataModel.ngrams)
        }
        return probabilities.filter { it.value < 0.0 }
    }

    internal fun computeSumOfNgramProbabilities(
        language: Language,
        ngrams: Set<Ngram>
    ): Double {
        val probabilities = mutableListOf<Double>()

        for (ngram in ngrams) {
            for (elem in ngram.rangeOfLowerOrderNgrams()) {
                val probability = lookUpNgramProbability(language, elem)
                if (probability > 0) {
                    probabilities.add(probability)
                    break
                }
            }
        }
        return probabilities.sumByDouble { ln(it) }
    }

    internal fun lookUpNgramProbability(
        language: Language,
        ngram: Ngram
    ): Double {
        val languageModels = when (ngram.value.length) {
            5 -> fivegramLanguageModels
            4 -> quadrigramLanguageModels
            3 -> trigramLanguageModels
            2 -> bigramLanguageModels
            1 -> unigramLanguageModels
            0 -> throw IllegalArgumentException("Zerogram detected")
            else -> throw IllegalArgumentException("unsupported ngram length detected: ${ngram.value.length}")
        }

        return languageModels.getValue(language).value.getRelativeFrequency(ngram)
    }

    internal fun loadLanguageModel(
        language: Language,
        ngramLength: Int
    ): TrainingDataLanguageModel {
        val fileName = "${Ngram.getNgramNameByLength(ngramLength)}s.json"
        val filePath = "/language-models/${language.isoCode639_1}/$fileName"
        val inputStream = LanguageDetector::class.java.getResourceAsStream(filePath)
        val jsonContent = inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        return TrainingDataLanguageModel.fromJson(jsonContent)
    }

    internal fun loadLanguageModels(ngramLength: Int): MutableMap<Language, Lazy<TrainingDataLanguageModel>> {
        val languageModels = hashMapOf<Language, Lazy<TrainingDataLanguageModel>>()
        for (language in languages) {
            languageModels[language] = lazy { loadLanguageModel(language, ngramLength) }
        }
        return languageModels
    }

    override fun equals(other: Any?) = when {
        this === other -> true
        other !is LanguageDetector -> false
        languages != other.languages -> false
        minimumRelativeDistance != other.minimumRelativeDistance -> false
        else -> true
    }

    override fun hashCode() = 31 * languages.hashCode() + minimumRelativeDistance.hashCode()

    internal companion object {
        private val NO_LETTER = Regex("^[^\\p{L}]+$")

        private val CHARS_TO_LANGUAGES_MAPPING = hashMapOf(

            "Îî" to setOf(FRENCH),
            "Ññ" to setOf(SPANISH),

            "Ûû" to setOf(FRENCH),

            "Ëë" to setOf(FRENCH),
            "ÈèÙù" to setOf(FRENCH),
            "Êê" to setOf(FRENCH),
            "Ôô" to setOf(FRENCH),

            "Àà" to setOf(FRENCH),

            "Üü" to setOf(SPANISH),

            "Çç" to setOf(FRENCH),

            "Óó" to setOf(SPANISH),
            "ÁáÍíÚú" to setOf(SPANISH),

            "Éé" to setOf(FRENCH, SPANISH)
        )
    }
}
