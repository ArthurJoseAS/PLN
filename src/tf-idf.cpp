
#include <cctype>
#include <cmath>
#include <vector>
#include <optional>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include "stopwords.h"

struct Review {
    std::map<std::string, int> word_counts;
    int label;
    int total_words = 0;
};

std::map<std::string, int> vocabulario;
std::map<std::string, int> document_frequency;
std::vector<Review> reviews;

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> words;
    std::string current;

    for (unsigned char c : s) {
        if (std::isalpha(c)) {
            current += std::tolower(c);
        } else {
            if (!current.empty() && !stopwords.count(current)) {
                words.push_back(current);
            }
            current.clear();
        }
    }

    if (!current.empty() && !stopwords.count(current)) {
        words.push_back(current);
    }

    return words;
}

std::optional<std::ifstream> open_file(char* file_arg) {
    std::ifstream f(file_arg);

    if (!f.is_open()) {
        return {};
    }

    return f;
}

// TF = frequência da palavra / quantidade total de palavras
double calculate_tf(const Review& review, const std::string& word) {
    auto it = review.word_counts.find(word);

    if (it == review.word_counts.end() || review.total_words == 0) {
        return 0.0;
    }

    return static_cast<double>(it->second) / review.total_words;
}

// IDF = log(N / DF)
double calculate_idf(const std::string& word) {
    return std::log(
        static_cast<double>(reviews.size()) /
        document_frequency[word]
    );
}

// TF-IDF = TF * IDF
double calculate_tfidf(const Review& review, const std::string& word) {
    return calculate_tf(review, word) * calculate_idf(word);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Adicione um arquivo para abrir\n";
        return 1;
    }

    auto arquivo = open_file(argv[1]);

    if (!arquivo) {
        std::cerr << "Arquivo invalido\n";
        return 1;
    }

    std::string line;

    // Read reviews and build vocabulary / document frequencies
    while (std::getline(arquivo.value(), line)) {
        size_t separator = line.rfind('\t');

        if (separator == std::string::npos ||
            separator + 1 >= line.size()) {
            continue;
        }

        std::string text = line.substr(0, separator);
        int label = line[separator + 1] - '0';

        Review review;
        review.label = label;

        for (const auto& word : tokenize(text)) {
            review.word_counts[word]++;
            review.total_words++;
            vocabulario[word] = 1;
        }

        // Each word contributes only once to document frequency
        for (const auto& [word, count] : review.word_counts) {
            document_frequency[word]++;
        }

        reviews.push_back(std::move(review));
    }

    std::ofstream output("Imdb_TF-IDF.csv");

    if (!output.is_open()) {
        std::cerr << "Erro ao criar arquivo de saida\n";
        return 1;
    }

    // CSV header
    for (const auto& [word, _] : vocabulario) {
        output << word << ",";
    }

    output << "label\n";

    // Generate TF-IDF vectors
    for (const auto& review : reviews) {
        for (const auto& [word, _] : vocabulario) {
            output << calculate_tfidf(review, word) << ",";
        }

        output << review.label << "\n";
    }
}
